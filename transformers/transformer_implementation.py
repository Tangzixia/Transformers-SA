"""
Transformer 架构完整实现

这个脚本包含了完整的 Transformer 模型实现，
包括所有核心组件和实战训练示例。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

# ============================================================================
# 1. ScaledDotProductAttention（缩放点积注意力）
# ============================================================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            Q (Query): [batch, num_heads, seq_len_q, d_k]
            K (Key):   [batch, num_heads, seq_len_k, d_k]
            V (Value): [batch, num_heads, seq_len_v, d_v]
            mask: 掩码矩阵
        
        返回:
            output: 注意力加权的值
            attention_weights: 注意力权重
        """
        # 计算相似度分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


# ============================================================================
# 2. MultiHeadAttention（多头注意力）
# ============================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model 必须能被 num_heads 整除'
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = Q.shape[0]
        
        # 线性投影并分割为多个头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # 合并多个头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终的线性投影
        output = self.W_o(output)
        
        return output, attention_weights


# ============================================================================
# 3. FeedForwardNetwork（前馈网络）
# ============================================================================

class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ============================================================================
# 4. PositionalEncoding（位置编码）
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# 5. EncoderLayer 和 DecoderLayer
# ============================================================================

class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 自注意力
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, attn_weights


class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 自注意力（因果）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 交叉注意力
        cross_attn_output, cross_attn_weights = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x, self_attn_weights, cross_attn_weights


# ============================================================================
# 6. Encoder 和 Decoder 堆栈
# ============================================================================

class Encoder(nn.Module):
    """编码器堆栈"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        x = self.norm(x)
        return x, attn_weights_list


class Decoder(nn.Module):
    """解码器堆栈"""
    
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list, list]:
        self_attn_weights_list = []
        cross_attn_weights_list = []
        for layer in self.layers:
            x, self_attn_w, cross_attn_w = layer(x, encoder_output, causal_mask)
            self_attn_weights_list.append(self_attn_w)
            cross_attn_weights_list.append(cross_attn_w)
        x = self.norm(x)
        return x, self_attn_weights_list, cross_attn_weights_list


# ============================================================================
# 7. 完整 Transformer 模型
# ============================================================================

class Transformer(nn.Module):
    """完整的 Transformer 模型"""
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8, num_layers: int = 6,
                 d_ff: int = 2048, max_seq_length: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.positional_encoding(src_embedded)
        encoder_output, encoder_attn = self.encoder(src_embedded, src_mask)
        
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.positional_encoding(tgt_embedded)
        decoder_output, decoder_self_attn, decoder_cross_attn = self.decoder(tgt_embedded, encoder_output, tgt_mask)
        
        logits = self.output_projection(decoder_output)
        return logits, {'encoder_attn': encoder_attn, 'decoder_self_attn': decoder_self_attn, 'decoder_cross_attn': decoder_cross_attn}


# ============================================================================
# 8. 工具函数
# ============================================================================

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """创建因果掩码"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_copy_dataset(num_samples=100, max_seq_len=10, vocab_size=20):
    """创建复制任务数据集"""
    src_seqs = []
    tgt_seqs = []
    for _ in range(num_samples):
        seq_len = np.random.randint(3, max_seq_len + 1)
        seq = np.random.randint(1, vocab_size, seq_len)
        src_seqs.append(seq)
        tgt_seqs.append(seq)
    return src_seqs, tgt_seqs


# ============================================================================
# 9. 示例：训练
# ============================================================================

if __name__ == "__main__":
    # 设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 20
    src_seqs, tgt_seqs = create_copy_dataset(num_samples=100, vocab_size=vocab_size)
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Transformer 模型创建完成")
