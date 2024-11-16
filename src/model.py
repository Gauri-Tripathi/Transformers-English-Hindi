import torch 
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_seq_len = x.shape[-2]
        if actual_seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({actual_seq_len}) exceeds maximum sequence length "
                f"({self.max_seq_len}) for RotaryEmbedding."
            )
            
        cos = self.cos_cached[:actual_seq_len, ...].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:actual_seq_len, ...].unsqueeze(0).unsqueeze(0)
        
        return x * cos + self._rotate_half(x) * sin

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float)-> None:
        super().__init__()  
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.shape[1], :].requires_grad_(False)
        x = x + pe
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)).clamp(-100, 100)))

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)  
        self.norm = LayerNormalization(features)
        
    def forward(self, x, sublayer_callable_module):
        normed_x = self.norm(x)
        sublayer_output_package = sublayer_callable_module(normed_x)
        
        if isinstance(sublayer_output_package, tuple):
            actual_sublayer_output, scores = sublayer_output_package
            return x + self.dropout(actual_sublayer_output), scores
        else:
            actual_sublayer_output = sublayer_output_package
            return x + self.dropout(actual_sublayer_output), None

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, use_rope: bool = False, max_seq_len: int = 0) -> None:
        super().__init__()
        self.d_model = d_model 
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by head"
        
        self.d_k = d_model // h 
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.use_rope = use_rope
        if self.use_rope:
            if max_seq_len <= 0:
                raise ValueError("max_seq_len must be positive when use_rope is True for MultiHeadAttention")
            self.rope = RotaryEmbedding(self.d_k, max_seq_len)
        else:
            self.rope = None

        self.use_flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, q_in, k_in, v_in, mask, return_attention_scores: bool = False):
        batch_size = q_in.shape[0]
        
        dtype = self.w_q.weight.dtype
        q = q_in.to(dtype)
        k = k_in.to(dtype)
        v = v_in.to(dtype)
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        if self.rope:
            query = self.rope(query)
            key = self.rope(key)

        attention_scores_to_return = None
        if self.use_flash_attention:
            if mask is not None:
                if len(mask.shape) == 3:
                    mask = mask.unsqueeze(1)
                
                mask = mask.to(dtype=torch.bool)
                if mask.shape[1] == 1 and self.h > 1:
                    mask = mask.expand(-1, self.h, -1, -1)
            
            x = torch.nn.functional.scaled_dot_product_attention(
                query, key, value,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            scores_raw = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores_raw = scores_raw.masked_fill(mask == 0, float('-inf'))
            attention_scores_softmaxed = torch.softmax(scores_raw, dim=-1)
            attention_scores_to_return = self.dropout(attention_scores_softmaxed)
            x = torch.matmul(attention_scores_to_return, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.w_o(x)
        
        if return_attention_scores:
            return x, attention_scores_to_return
        return x

class EncoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_module:MultiHeadAttention, feed_forward_module:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_module = self_attention_module
        self.feed_forward_module = feed_forward_module
        self.residual_connection_1 = ResidualConnection(features, dropout)
        self.residual_connection_2 = ResidualConnection(features, dropout)

    def forward(self, x, src_mask, return_attention_scores: bool = False):
        mha_sublayer_callable = lambda normed_x: self.self_attention_module(normed_x, normed_x, normed_x, src_mask, return_attention_scores)
        x_after_mha, self_attn_scores = self.residual_connection_1(x, mha_sublayer_callable)
        
        ff_sublayer_callable = lambda normed_x: self.feed_forward_module(normed_x)
        x_after_ff, _ = self.residual_connection_2(x_after_mha, ff_sublayer_callable)
        
        if return_attention_scores:
            return x_after_ff, self_attn_scores
        return x_after_ff
    
class Encoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.final_norm = LayerNormalization(features)

    def forward(self, x, mask, return_attention_scores: bool = False):
        all_layer_attention_scores = [] if return_attention_scores else None
        
        for layer in self.layers:
            layer_output_package = layer(x, mask, return_attention_scores)
            
            if return_attention_scores:
                x, scores_for_layer = layer_output_package
                all_layer_attention_scores.append(scores_for_layer)
            else:
                x = layer_output_package
        
        x = self.final_norm(x)
        
        if return_attention_scores:
            return x, all_layer_attention_scores
        return x

class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_module:MultiHeadAttention, cross_attention_module:MultiHeadAttention, feed_forward_module:FeedForwardBlock, dropout:float) -> None:
        super().__init__()
        self.self_attention_module = self_attention_module
        self.cross_attention_module = cross_attention_module
        self.feed_forward_module = feed_forward_module
        self.residual_connection_1 = ResidualConnection(features, dropout)
        self.residual_connection_2 = ResidualConnection(features, dropout)
        self.residual_connection_3 = ResidualConnection(features, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention_scores: bool = False):
        self_mha_callable = lambda normed_x: self.self_attention_module(normed_x, normed_x, normed_x, tgt_mask, return_attention_scores)
        x_after_self_mha, self_attn_scores = self.residual_connection_1(x, self_mha_callable)
        
        cross_mha_callable = lambda normed_q_input: self.cross_attention_module(normed_q_input, encoder_output, encoder_output, src_mask, return_attention_scores)
        x_after_cross_mha, cross_attn_scores = self.residual_connection_2(x_after_self_mha, cross_mha_callable)
        
        ff_callable = lambda normed_x: self.feed_forward_module(normed_x)
        x_after_ff, _ = self.residual_connection_3(x_after_cross_mha, ff_callable)
        
        if return_attention_scores:
            return x_after_ff, self_attn_scores, cross_attn_scores
        return x_after_ff

class Decoder(nn.Module):
    def __init__(self, features:int, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.final_norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask, return_attention_scores: bool = False):
        all_layer_self_attn_scores = [] if return_attention_scores else None
        all_layer_cross_attn_scores = [] if return_attention_scores else None
        
        for layer in self.layers:
            layer_output_package = layer(x, encoder_output, src_mask, tgt_mask, return_attention_scores)
            
            if return_attention_scores:
                x, self_scores_for_layer, cross_scores_for_layer = layer_output_package
                all_layer_self_attn_scores.append(self_scores_for_layer)
                all_layer_cross_attn_scores.append(cross_scores_for_layer)
            else:
                x = layer_output_package
        
        x = self.final_norm(x)
        
        if return_attention_scores:
            return x, all_layer_self_attn_scores, all_layer_cross_attn_scores
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder_module:Encoder, decoder_module:Decoder, src_embed_module:InputEmbeddings, tgt_embed_module:InputEmbeddings, 
                 src_pos_module:PositionalEncoding, tgt_pos_module:PositionalEncoding, projection_module:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder_module
        self.decoder = decoder_module
        self.src_embed = src_embed_module
        self.tgt_embed = tgt_embed_module
        self.src_pos = src_pos_module
        self.tgt_pos = tgt_pos_module
        self.projection_layer = projection_module
        self.gradient_checkpointing = False
    
    def forward(self, src, tgt, src_mask, tgt_mask, return_attention_scores: bool = False):
        if self.gradient_checkpointing and self.training and not return_attention_scores:
            return torch.utils.checkpoint.checkpoint(self._forward_for_checkpoint, src, tgt, src_mask, tgt_mask, use_reentrant=False)
        return self._internal_forward(src, tgt, src_mask, tgt_mask, return_attention_scores)

    def _forward_for_checkpoint(self, src, tgt, src_mask, tgt_mask):
        return self._internal_forward(src, tgt, src_mask, tgt_mask, return_attention_scores=False)
    
    def _internal_forward(self, src, tgt, src_mask, tgt_mask, return_attention_scores: bool = False):
        encode_output_package = self.encode(src, src_mask, return_attention_scores)
        
        if return_attention_scores:
            encoder_output, encoder_attns = encode_output_package
        else:
            encoder_output = encode_output_package
            encoder_attns = None

        decode_output_package = self.decode(encoder_output, src_mask, tgt, tgt_mask, return_attention_scores)
        
        if return_attention_scores:
            decoder_output, decoder_self_attns, decoder_cross_attns = decode_output_package
        else:
            decoder_output = decode_output_package
            decoder_self_attns, decoder_cross_attns = None, None
            
        proj_output = self.project(decoder_output)
        
        if return_attention_scores:
            return proj_output, {
                "encoder_self_attention": encoder_attns,
                "decoder_self_attention": decoder_self_attns,
                "decoder_cross_attention": decoder_cross_attns
            }
        return proj_output

    def encode(self, src, src_mask, return_attention_scores: bool = False):
        src_embedded = self.src_embed(src)
        if self.src_pos:
            src_embedded = self.src_pos(src_embedded)
        return self.encoder(src_embedded, src_mask, return_attention_scores)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask, return_attention_scores: bool = False):
        tgt_embedded = self.tgt_embed(tgt)
        if self.tgt_pos:
            tgt_embedded = self.tgt_pos(tgt_embedded)
        return self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask, return_attention_scores)
        
    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len:int, 
                     d_model:int=512, num_encoder_layers:int=6, num_decoder_layers:int=6,
                     num_heads:int=8, dropout_rate:float=0.1, d_ff:int=2048,
                     positional_encoding_type:str = 'additive') -> Transformer:

    src_embed_module = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed_module = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos_module = None
    tgt_pos_module = None
    use_rope_for_mha = False

    if positional_encoding_type == 'additive':
        src_pos_module = PositionalEncoding(d_model, src_seq_len, dropout_rate)
        tgt_pos_module = PositionalEncoding(d_model, tgt_seq_len, dropout_rate)
    elif positional_encoding_type == 'rope':
        use_rope_for_mha = True
    else:
        raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}")

    encoder_layer_list = []
    for _ in range(num_encoder_layers):
        self_attn_module = MultiHeadAttention(d_model, num_heads, dropout_rate, use_rope=use_rope_for_mha, max_seq_len=src_seq_len)
        ff_module = FeedForwardBlock(d_model, d_ff, dropout_rate)
        encoder_block_instance = EncoderBlock(d_model, self_attn_module, ff_module, dropout_rate)
        encoder_layer_list.append(encoder_block_instance)

    decoder_layer_list = []    
    for _ in range(num_decoder_layers):
        self_attn_module_dec = MultiHeadAttention(d_model, num_heads, dropout_rate, use_rope=use_rope_for_mha, max_seq_len=tgt_seq_len)
        cross_attn_module_dec = MultiHeadAttention(d_model, num_heads, dropout_rate, use_rope=False, max_seq_len=tgt_seq_len)
        ff_module_dec = FeedForwardBlock(d_model, d_ff, dropout_rate)
        decoder_block_instance = DecoderBlock(d_model, self_attn_module_dec, cross_attn_module_dec, ff_module_dec, dropout_rate)
        decoder_layer_list.append(decoder_block_instance)

    encoder_module_inst = Encoder(d_model, nn.ModuleList(encoder_layer_list))
    decoder_module_inst = Decoder(d_model, nn.ModuleList(decoder_layer_list))

    projection_module_inst = ProjectionLayer(d_model, tgt_vocab_size)

    transformer_instance = Transformer(
        encoder_module_inst, decoder_module_inst, 
        src_embed_module, tgt_embed_module, 
        src_pos_module, tgt_pos_module, 
        projection_module_inst
    )

    for p in transformer_instance.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    return transformer_instance

