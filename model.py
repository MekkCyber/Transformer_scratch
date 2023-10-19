import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        out = self.embed(x)*math.sqrt(self.embed_dim)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length, dropout):
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        self.pe = torch.zeros(self.seq_length, self.d_model)

        position = torch.arange(0,seq_length, dtype=torch.float32).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/self.d_model))

        self.pe[:,0::2] = torch.sin(torch.matmul(position,denominator))
        self.pe[:,1::2] = torch.cos(torch.matmul(position,denominator))
        # add batch dimension
        self.pe.unsqueeze(0)

        self.register_buffer('pe',self.pe)

    def forward(self, x) : 
        x = x + self.pe.requires_grad(False)
        return self.droupout(x)

class LayerNormalization(nn.Module) : 
    def __init__(self, epsilon==10**-6) : 
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x) : 
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha* (x-mean) / (std+self.epsilon) + self.bias
    

class FeedForward(nn.Module) : 
    def __init__(self, d_model, d_ff, dropout) : 
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x) : 
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model, h, dropout) : 
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model % h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask, dropout) : 
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / torch.sqrt()
        if mask : 
            attention_scores.fill_masked_(mask==0, 1e-9)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        if dropout is not None : 
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    def forward(self, q, k, v, mask) : 
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[2], x.shape[1]*x.shape[-1])

        return self.w_o(x)
    


class ResidualConnection(nn.Module) : 
    def __init__(self, dropout) : 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer) : 
        x = x + self.dropout(sublayer(self.norm(x)))
        return x
    

class EncoderBlock(nn.Module) : 
    def __init__(self, self_attention : MultiHeadAttention, ff_block : FeedForward, dropout) : 
        super().__init__()
        self.self_attention = self_attention
        self.ff_block = ff_block
        self.resid1 = ResidualConnection(dropout)
        self.resid2 = ResidualConnection(dropout)

    def forward(self, x, mask) : 
        x = self.resid1(x,lambda x : self.self_attention(x,x,x,mask))
        x = self.resid2(x, lambda x : self.ff_block(x))
        return x
    

class Encoder(nn.Module) : 
    def __init__(self, layers) : 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask) : 
        for layer in self.layers : 
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module) : 
    def __init__(self, self_attention : MultiHeadAttention, cross_attention : MultiHeadAttention, ff_block : FeedForward, dropout) : 
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ff_block = ff_block
        self.resid = nn.Module([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, enc_out, src_mask, tg_mask) : 
        x = self.resid[0](x,lambda x : self.self_attention(x,x,x,tg_mask))
        x = self.resid[1](x, lambda x : self.cross_attention(x, enc_out, enc_out, src_mask))
        x = self.resid[2](x, self.ff_block)
        return x
    

class Decoder(nn.Module) : 
    def __init__(self, layers) : 
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, enc_out, src_mask, tg_mask) : 
        for layer in self.layers : 
            x = layer(x, enc_out, src_mask, tg_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module) : 
    def __init__(self, d_model, vocab_size) : 
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) 
    def forward(self, x) : 
        x = self.proj(x) 
        return torch.log_softmax(x, dim=-1)
    
