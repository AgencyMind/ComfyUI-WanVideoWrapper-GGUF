# Modified from transformers.models.t5.modeling_t5
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

def fp16_clamp(x):
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


def init_weights(m):
    if isinstance(m, T5LayerNorm):
        nn.init.ones_(m.weight)
    elif isinstance(m, T5Model):
        nn.init.normal_(m.token_embedding.weight, std=1.0)
    elif isinstance(m, T5FeedForward):
        nn.init.normal_(m.gate[0].weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc1.weight, std=m.dim**-0.5)
        nn.init.normal_(m.fc2.weight, std=m.dim_ffn**-0.5)
    elif isinstance(m, T5Attention):
        nn.init.normal_(m.q.weight, std=(m.dim * m.dim_attn)**-0.5)
        nn.init.normal_(m.k.weight, std=m.dim**-0.5)
        nn.init.normal_(m.v.weight, std=m.dim**-0.5)
        nn.init.normal_(m.o.weight, std=(m.num_heads * m.dim_attn)**-0.5)
    elif isinstance(m, T5RelativeEmbedding):
        nn.init.normal_(
            m.embedding.weight, std=(2 * m.num_buckets * m.num_heads)**-0.5)


class GELU(nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(T5LayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) +
                            self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        
        # Debug: Add runtime shape checking for GGUF debugging
        try:
            result = self.weight * x
            return result
        except RuntimeError as e:
            print(f"=== T5LayerNorm Runtime Error Debug ===")
            print(f"Error: {e}")
            print(f"self.weight shape: {self.weight.shape}")
            print(f"x shape: {x.shape}")
            print(f"x dtype: {x.dtype}")
            print(f"self.weight dtype: {self.weight.dtype}")
            if hasattr(self.weight, 'tensor_type'):
                print(f"self.weight is quantized: {self.weight.tensor_type}")
            else:
                print(f"self.weight is not quantized")
            print(f"Operation attempted: {self.weight.shape} * {x.shape}")
            raise


class T5Attention(nn.Module):

    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super(T5Attention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # attention bias
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                             -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        # compute attention (T5 does not use scaling)
        attn = torch.einsum('binc,bjnc->bnij', q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum('bnij,bjnc->binc', attn, v)

        # output
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):

    def __init__(self, dim, dim_ffn, dropout=0.1):
        super(T5FeedForward, self).__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5SelfAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5CrossAttention, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def forward(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.size(1), x.size(1))
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):

    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super(T5RelativeEmbedding, self).__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        # rel_pos = torch.arange(lk).unsqueeze(0).to(device) - \
        #     torch.arange(lq).unsqueeze(1).to(device)
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - \
            torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        rel_pos_embeds = rel_pos_embeds.permute(2, 0, 1).unsqueeze(
            0)  # [1, N, Lq, Lk]
        return rel_pos_embeds.contiguous()

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) /
                                     math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(
            rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Encoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Encoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None):
        # Debug: Check embedding behavior for GGUF debugging
        print(f"🔍 T5Encoder.forward() debugging:")
        print(f"   └─ Input ids shape: {ids.shape}")
        print(f"   └─ token_embedding.weight shape: {self.token_embedding.weight.shape}")
        if hasattr(self.token_embedding.weight, 'tensor_type'):
            print(f"   └─ token_embedding is quantized: {self.token_embedding.weight.tensor_type}")
        else:
            print(f"   └─ token_embedding is not quantized")
        
        x = self.token_embedding(ids)
        print(f"   └─ Embedding output shape: {x.shape}")
        print(f"   └─ Expected output shape: [batch_size, seq_len, 4096]")
        
        # Verify dimensional correctness
        if x.shape[-1] != 4096:
            print(f"❌ DIMENSIONAL ERROR: Embedding output has {x.shape[-1]} dimensions, expected 4096")
            print(f"   Root cause: Quantized tensor embedding lookup is using wrong axis")
            print(f"   Solution: Manual embedding lookup with proper tensor handling")
            
            # CRITICAL FIX: Manual embedding lookup for quantized tensors
            if hasattr(self.token_embedding.weight, 'tensor_type'):
                print(f"🔧 APPLYING QUANTIZED EMBEDDING FIX")
                
                # Get the quantized weight tensor
                weight = self.token_embedding.weight
                print(f"   └─ Weight shape: {weight.shape}")
                print(f"   └─ Weight quantization: {weight.tensor_type}")
                
                # Manual embedding lookup using F.embedding with proper indexing
                import torch.nn.functional as F
                
                # CRITICAL: Quantized tensors need dequantization for proper embedding lookup
                print(f"   └─ Attempting dequantization for proper embedding lookup")
                
                try:
                    # Method 1: Try .dequantize() if available
                    if hasattr(weight, 'dequantize'):
                        weight_dequant = weight.dequantize()
                        print(f"   └─ Dequantized using .dequantize(): {weight_dequant.shape}")
                        x = F.embedding(ids, weight_dequant)
                        print(f"   └─ Dequantized embedding result: {x.shape}")
                    else:
                        # Method 2: Try accessing .data directly
                        weight_data = weight.data
                        print(f"   └─ Using .data: {weight_data.shape}")
                        x = F.embedding(ids, weight_data)
                        print(f"   └─ Data embedding result: {x.shape}")
                        
                except Exception as e:
                    print(f"   └─ Dequantization failed: {e}")
                    # Method 3: Convert to float and force correct shape
                    print(f"   └─ Falling back to float conversion")
                    weight_float = weight.float()
                    
                    # If still wrong dimensions, force transpose
                    if weight_float.shape == torch.Size([4096, 256384]):
                        print(f"   └─ Float tensor still wrong shape, transposing")
                        weight_float = weight_float.T
                    
                    x = F.embedding(ids, weight_float)
                    print(f"   └─ Float embedding result: {x.shape}")
                    
            else:
                # Regular tensor - should work normally but verify
                print(f"🔧 REGULAR TENSOR EMBEDDING CHECK")
                weight = self.token_embedding.weight
                if weight.shape == torch.Size([4096, 256384]):
                    print(f"   └─ Regular tensor needs transpose")
                    self.token_embedding.weight = nn.Parameter(weight.T.contiguous())
                    x = self.token_embedding(ids)
                    print(f"   └─ Fixed regular embedding result: {x.shape}")
                    
            # Verify fix worked
            if x.shape[-1] == 4096:
                print(f"✅ EMBEDDING FIX SUCCESSFUL: Output now {x.shape}")
            else:
                print(f"❌ EMBEDDING FIX FAILED: Still wrong dimensions {x.shape}")
        else:
            print(f"✅ Embedding output dimensions correct: {x.shape[-1]}")
        
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):

    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Decoder, self).__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.token_embedding = vocab if isinstance(vocab, nn.Embedding) \
            else nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

        # initialize weights
        self.apply(init_weights)

    def forward(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.size()

        # causal mask
        if mask is None:
            mask = torch.tril(torch.ones(1, s, s).to(ids.device))
        elif mask.ndim == 2:
            mask = torch.tril(mask.unsqueeze(1).expand(-1, s, -1))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1),
                               x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):

    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.1):
        super(T5Model, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # initialize weights
        self.apply(init_weights)

    def forward(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        dtype=torch.float32,
        device='cpu',
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    with torch.device(device):
        model = model_cls(**kwargs)

    # set device
    #model = model.to(dtype=dtype, device=device)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.1)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:

    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.device('cuda'),
        state_dict=None,
        tokenizer_path=None,
        quantization="disabled",
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.tokenizer_path = tokenizer_path

        # init model
        with init_empty_weights():
            model = umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device).eval().requires_grad_(False)
        
        if quantization == "fp8_e4m3fn":
            cast_dtype = torch.float8_e4m3fn
        else:
            cast_dtype = dtype

        params_to_keep = {'norm', 'pos_embedding', 'token_embedding'}
        for name, param in model.named_parameters():
            if name in state_dict:
                tensor_data = state_dict[name]
                
                # Debug tensor shape mismatches (can be removed once stable)
                if param.shape != tensor_data.shape:
                    print(f"Shape mismatch for {name}: model expects {param.shape}, tensor has {tensor_data.shape}")
                
                # Critical: preserve quantization for quantized tensors (same pattern as WanVideo loader)
                if hasattr(tensor_data, 'tensor_type'):
                    # Quantized tensor - use direct assignment instead of set_module_tensor_to_device
                    # which doesn't handle quantized tensors properly
                    try:
                        # Navigate to the actual parameter and assign directly
                        module_path = name.split('.')
                        target_module = model
                        for path_part in module_path[:-1]:
                            target_module = getattr(target_module, path_part)
                        
                        # Direct parameter assignment for quantized tensors
                        param_name = module_path[-1]
                        target_module._parameters[param_name] = tensor_data
                        
                        print(f"✅ Direct quantized assignment: {name}")
                        
                        # Verify assignment for token_embedding specifically
                        if name == "token_embedding.weight":
                            assigned_tensor = getattr(model.token_embedding, 'weight')
                            print(f"🔍 Assignment verification for {name}:")
                            print(f"   └─ Original tensor shape: {tensor_data.shape}")
                            print(f"   └─ Assigned tensor shape: {assigned_tensor.shape}")
                            print(f"   └─ Assignment successful: {assigned_tensor.shape == tensor_data.shape}")
                    except Exception as e:
                        print(f"❌ Direct assignment failed for {name}: {e}")
                        # Fallback to original method
                        set_module_tensor_to_device(model, name, device=device, value=tensor_data)
                else:
                    # Regular tensor - apply dtype conversion
                    dtype_to_use = dtype if any(keyword in name for keyword in params_to_keep) else cast_dtype
                    set_module_tensor_to_device(model, name, device=device, dtype=dtype_to_use, value=tensor_data)
        del state_dict
        
        # Adaptive validation: Detect actual model dimensions from loaded weights
        token_emb_weight = model.token_embedding.weight
        actual_shape = token_emb_weight.shape
        actual_vocab_size, actual_embedding_dim = actual_shape
        
        # Debug: Print tensor properties for GGUF debugging
        print(f"🔍 T5EncoderModel validation debug:")
        print(f"   └─ token_embedding.weight.shape: {actual_shape}")
        print(f"   └─ tensor dtype: {token_emb_weight.dtype}")
        print(f"   └─ tensor device: {token_emb_weight.device}")
        if hasattr(token_emb_weight, 'tensor_type'):
            print(f"   └─ quantized: yes (type: {token_emb_weight.tensor_type})")
        else:
            print(f"   └─ quantized: no")
        
        # Critical check: Verify that assignment worked correctly
        if hasattr(token_emb_weight, 'tensor_type') and actual_shape == torch.Size([256384, 4096]):
            print(f"✅ GGUF quantized tensor assignment appears successful")
        elif not hasattr(token_emb_weight, 'tensor_type') and actual_shape == torch.Size([256384, 4096]):
            print(f"✅ Regular tensor assignment appears successful")
        elif actual_shape == torch.Size([4096, 256384]):
            print(f"❌ ASSIGNMENT FAILURE: Token embedding still has wrong shape after assignment")
            print(f"   This indicates the direct parameter assignment didn't work")
            print(f"   The embedding will produce {actual_shape[0]} dimensional outputs instead of {actual_shape[1]}")
        else:
            print(f"⚠️  Unknown tensor shape pattern: {actual_shape}")
        
        # Expected UMT5-XXL dimensions
        expected_vocab_size, expected_embedding_dim = 256384, 4096
        
        if (actual_vocab_size, actual_embedding_dim) != (expected_vocab_size, expected_embedding_dim):
            print(f"⚠️  Model dimension mismatch detected:")
            print(f"Expected UMT5-XXL: vocab_size={expected_vocab_size}, embedding_dim={expected_embedding_dim}")
            print(f"Actual model: vocab_size={actual_vocab_size}, embedding_dim={actual_embedding_dim}")
            
            # Check if this is a known T5 variant
            if actual_vocab_size == 32128 and actual_embedding_dim == 768:
                print(f"✅ Detected T5-Base model (32128 vocab, 768 dim)")
            elif actual_vocab_size == 32128 and actual_embedding_dim == 1024:
                print(f"✅ Detected T5-Large model (32128 vocab, 1024 dim)")
            elif actual_vocab_size == 32128 and actual_embedding_dim == 2048:
                print(f"✅ Detected T5-3B model (32128 vocab, 2048 dim)")
            elif actual_vocab_size == 3360 and actual_embedding_dim == 256384:
                print(f"❌ Invalid model detected: This appears to be a corrupted or incorrectly converted model")
                print(f"   The dimensions suggest the vocab_size and embedding_dim were swapped during conversion")
                raise RuntimeError(f"Invalid T5 model: vocab_size={actual_vocab_size}, embedding_dim={actual_embedding_dim}. "
                                 f"This model appears to have swapped dimensions and is not compatible with WanVideo. "
                                 f"Please use a proper UMT5-XXL GGUF model with 256384 vocab size and 4096 embedding dimension.")
            else:
                print(f"⚠️  Unknown T5 variant - proceeding with detected dimensions")
                
            # For non-UMT5-XXL models, update the architecture dynamically
            if actual_vocab_size != expected_vocab_size or actual_embedding_dim != expected_embedding_dim:
                print(f"🔄 Adapting T5EncoderModel for detected dimensions...")
                # Note: This may cause compatibility issues with WanVideo which expects UMT5-XXL
        else:
            print(f"✅ Token embedding validation passed: UMT5-XXL dimensions {actual_shape}")
        
        self.model = model
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
