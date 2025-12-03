#!/usr/bin/env python3
"""
Transformer Attention Mechanisms Benchmark and Comparison
完整的MHA、MQA、GQA、MLA实现和对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import time
import pandas as pd

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ============================================================================
# 1. 注意力机制实现
# ============================================================================

class MultiHeadAttention(nn.Module):
    """标准多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class MultiQueryAttention(nn.Module):
    """多查询注意力"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        K = self.W_k(key).unsqueeze(1)
        K = K.expand(batch_size, self.num_heads, -1, -1)
        
        V = self.W_v(value).unsqueeze(1)
        V = V.expand(batch_size, self.num_heads, -1, -1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class GroupedQueryAttention(nn.Module):
    """分组查询注意力"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        kv_dim = (d_model // num_heads) * num_kv_heads
        self.W_k = nn.Linear(d_model, kv_dim)
        self.W_v = nn.Linear(d_model, kv_dim)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        kv_dim = self.d_k
        K = self.W_k(key).view(batch_size, -1, self.num_kv_heads, kv_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_kv_heads, kv_dim).transpose(1, 2)
        
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output, attention_weights


class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力"""
    
    def __init__(self, d_model: int, num_heads: int, latent_dim: int = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        if latent_dim is None:
            latent_dim = d_model // 2
        self.latent_dim = latent_dim
        self.d_l = latent_dim // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k_latent = nn.Linear(d_model, self.latent_dim)
        self.W_v_latent = nn.Linear(d_model, self.latent_dim)
        self.W_v_out = nn.Linear(self.latent_dim, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        K_latent = self.W_k_latent(key).view(batch_size, -1, self.num_heads, self.d_l).transpose(1, 2)
        V_latent = self.W_v_latent(value).view(batch_size, -1, self.num_heads, self.d_l).transpose(1, 2)
        
        Q_latent = Q[..., :self.d_l] if self.d_k >= self.d_l else Q
        
        scores = torch.matmul(Q_latent, K_latent.transpose(-2, -1)) / np.sqrt(self.d_l)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_latent = torch.matmul(attention_weights, V_latent)
        context_latent_flat = context_latent.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)
        context = self.W_v_out(context_latent_flat)
        
        output = self.W_o(context)
        
        return output, attention_weights


# ============================================================================
# 2. 测试配置
# ============================================================================

d_model = 512
num_heads = 8
batch_size = 2
seq_len = 256

print("=" * 70)
print("Transformer 注意力机制对比测试")
print("=" * 70)
print(f"配置: d_model={d_model}, num_heads={num_heads}, batch_size={batch_size}, seq_len={seq_len}\n")

# 创建模型
mha = MultiHeadAttention(d_model, num_heads).to(device)
mqa = MultiQueryAttention(d_model, num_heads).to(device)
gqa_4 = GroupedQueryAttention(d_model, num_heads, 4).to(device)
gqa_2 = GroupedQueryAttention(d_model, num_heads, 2).to(device)
mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim=256).to(device)

# 创建输入
x = torch.randn(batch_size, seq_len, d_model).to(device)

# ============================================================================
# 3. 参数数量对比
# ============================================================================

params_data = {
    'MHA': sum(p.numel() for p in mha.parameters()),
    'MQA': sum(p.numel() for p in mqa.parameters()),
    'GQA-4': sum(p.numel() for p in gqa_4.parameters()),
    'GQA-2': sum(p.numel() for p in gqa_2.parameters()),
    'MLA-256': sum(p.numel() for p in mla.parameters()),
}

print("参数数量对比:")
print("-" * 50)
for name, count in params_data.items():
    reduction = (1 - count / params_data['MHA']) * 100 if name != 'MHA' else 0
    reduction_str = f"(-{reduction:.1f}%)" if reduction > 0 else "基准"
    print(f"{name:12} {count:>10,} params  {reduction_str:>15}")

# ============================================================================
# 4. 输出一致性对比
# ============================================================================

print("\n输出一致性对比:")
print("-" * 50)

with torch.no_grad():
    out_mha, _ = mha(x, x, x)
    out_mqa, _ = mqa(x, x, x)
    out_gqa4, _ = gqa_4(x, x, x)
    out_gqa2, _ = gqa_2(x, x, x)
    out_mla, _ = mla(x, x, x)

outputs = {
    'MQA': out_mqa,
    'GQA-4': out_gqa4,
    'GQA-2': out_gqa2,
    'MLA-256': out_mla,
}

print(f"{'方法':12} {'L2差异':>12} {'相对L2':>12} {'余弦相似度':>14}")
print("-" * 50)

for name, output in outputs.items():
    l2_diff = torch.norm(output - out_mha).item()
    rel_l2 = l2_diff / torch.norm(out_mha).item()
    cos_sim = F.cosine_similarity(output.flatten().unsqueeze(0), out_mha.flatten().unsqueeze(0)).item()
    print(f"{name:12} {l2_diff:>12.6f} {rel_l2:>12.6f} {cos_sim:>14.6f}")

# ============================================================================
# 5. 推理速度基准
# ============================================================================

print("\n推理速度基准 (ms):")
print("-" * 50)

def benchmark(model, x, num_iters=50):
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(x, x, x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_iters):
            model(x, x, x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        return elapsed / num_iters * 1000

times = {
    'MHA': benchmark(mha, x),
    'MQA': benchmark(mqa, x),
    'GQA-4': benchmark(gqa_4, x),
    'GQA-2': benchmark(gqa_2, x),
    'MLA-256': benchmark(mla, x),
}

print(f"{'方法':12} {'时间(ms)':>12} {'加速比':>12}")
print("-" * 50)

for name, t in times.items():
    speedup = times['MHA'] / t
    speedup_str = f"{speedup:.2f}x"
    print(f"{name:12} {t:>12.4f} {speedup_str:>12}")

# ============================================================================
# 6. 可视化对比
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 参数对比
ax = axes[0, 0]
methods = list(params_data.keys())
params = list(params_data.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = ax.bar(methods, params, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('参数数量', fontsize=11)
ax.set_title('参数数量对比', fontsize=12, fontweight='bold')
ax.set_yscale('log')
for bar, value in zip(bars, params):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

# 参数减少百分比
ax = axes[0, 1]
reduction_methods = [m for m in methods if m != 'MHA']
reduction_values = [(1 - params_data[m] / params_data['MHA']) * 100 for m in reduction_methods]
bars = ax.bar(reduction_methods, reduction_values, color=colors[1:], alpha=0.7, edgecolor='black')
ax.set_ylabel('参数减少比例 (%)', fontsize=11)
ax.set_title('相对于MHA的参数减少', fontsize=12, fontweight='bold')
for bar, value in zip(bars, reduction_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

# 推理速度
ax = axes[1, 0]
time_values = [times[m] for m in methods]
bars = ax.bar(methods, time_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('推理时间 (ms)', fontsize=11)
ax.set_title('推理速度对比 (seq_len=256)', fontsize=12, fontweight='bold')
for bar, value in zip(bars, time_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}', ha='center', va='bottom', fontsize=9)

# 加速比
ax = axes[1, 1]
speedup_values = [times['MHA'] / times[m] for m in methods]
bars = ax.bar(methods, speedup_values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.set_ylabel('加速比 (倍数)', fontsize=11)
ax.set_title('相对于MHA的加速', fontsize=12, fontweight='bold')
for bar, value in zip(bars, speedup_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/tangzixia/Documents/Code/Transformers/attention_mechanisms_comparison.png', 
            dpi=150, bbox_inches='tight')
print("\n✓ 图表已保存到: attention_mechanisms_comparison.png")

# ============================================================================
# 7. 长序列测试
# ============================================================================

print("\n长序列测试 (不同序列长度的性能):")
print("-" * 70)

seq_lengths = [128, 256, 512, 1024]
results = {name: [] for name in methods}

with torch.no_grad():
    for seq_len_test in seq_lengths:
        x_test = torch.randn(batch_size, seq_len_test, d_model).to(device)
        
        t_mha = benchmark(mha, x_test, num_iters=20)
        t_mqa = benchmark(mqa, x_test, num_iters=20)
        t_gqa4 = benchmark(gqa_4, x_test, num_iters=20)
        t_gqa2 = benchmark(gqa_2, x_test, num_iters=20)
        t_mla = benchmark(mla, x_test, num_iters=20)
        
        results['MHA'].append(t_mha)
        results['MQA'].append(t_mqa)
        results['GQA-4'].append(t_gqa4)
        results['GQA-2'].append(t_gqa2)
        results['MLA-256'].append(t_mla)

print(f"{'Seq Len':>8}", end='')
for name in methods:
    print(f" {name:>12}", end='')
print()
print("-" * 70)

for i, seq_len_test in enumerate(seq_lengths):
    print(f"{seq_len_test:>8}", end='')
    for name in methods:
        print(f" {results[name][i]:>12.4f}", end='')
    print()

# 绘制长序列曲线
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for i, name in enumerate(methods):
    ax.plot(seq_lengths, results[name], marker='o', label=name, 
            color=colors[i], linewidth=2, markersize=8)
ax.set_xlabel('序列长度', fontsize=12)
ax.set_ylabel('推理时间 (ms)', fontsize=12)
ax.set_title('不同序列长度的推理速度', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/tangzixia/Documents/Code/Transformers/attention_long_sequence.png',
            dpi=150, bbox_inches='tight')
print("\n✓ 长序列图表已保存到: attention_long_sequence.png")

print("\n" + "=" * 70)
print("所有测试完成！")
print("=" * 70)
