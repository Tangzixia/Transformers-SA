"""
生成 Transformer 架构流程图
创建多个详细的架构可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_attention_mechanism():
    """绘制注意力机制流程图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, 'Scaled Dot-Product Attention 机制', 
            fontsize=18, weight='bold', ha='center')
    
    # 输入层
    input_boxes = [
        (1, 7.5, 'Query (Q)', 'lightblue'),
        (4, 7.5, 'Key (K)', 'lightgreen'),
        (7, 7.5, 'Value (V)', 'lightyellow'),
    ]
    
    for x, y, label, color in input_boxes:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, weight='bold')
    
    # 操作 1: Q × K^T
    ax.annotate('', xy=(2.5, 6.5), xytext=(1.3, 7.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(2.5, 6.5), xytext=(3.7, 7.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    box1 = FancyBboxPatch((1.8, 6.2), 1.4, 0.6, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(box1)
    ax.text(2.5, 6.5, 'Q × K^T', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(2.5, 6.05, '(seq_len, seq_len)', ha='center', va='center', fontsize=8, style='italic')
    
    # 操作 2: / √d_k
    ax.annotate('', xy=(2.5, 5.3), xytext=(2.5, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    box2 = FancyBboxPatch((1.8, 4.8), 1.4, 0.5, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(box2)
    ax.text(2.5, 5.05, '÷ √d_k', ha='center', va='center', fontsize=10, weight='bold')
    
    # 操作 3: Softmax
    ax.annotate('', xy=(2.5, 4.0), xytext=(2.5, 4.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    box3 = FancyBboxPatch((1.8, 3.5), 1.4, 0.5, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(box3)
    ax.text(2.5, 3.75, 'Softmax', ha='center', va='center', fontsize=10, weight='bold')
    
    # Mask (可选)
    ax.annotate('', xy=(1.8, 3.75), xytext=(0.8, 3.75),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='dashed'))
    ax.text(0.5, 3.75, 'Mask\n(可选)', ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', linestyle='dashed'))
    
    # 操作 4: 乘以 V
    ax.annotate('', xy=(5, 2.5), xytext=(2.5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(5, 2.5), xytext=(7, 7.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    box4 = FancyBboxPatch((4.2, 2.0), 1.6, 0.5, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(box4)
    ax.text(5, 2.25, 'Attention(Q,K,V)', ha='center', va='center', fontsize=10, weight='bold')
    
    # 输出
    ax.annotate('', xy=(5, 1.0), xytext=(5, 2.0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    output_box = FancyBboxPatch((4.2, 0.3), 1.6, 0.7, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.67, 'Output', ha='center', va='center', fontsize=11, weight='bold')
    
    # 公式框
    formula_text = r'$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$'
    ax.text(5, 8.7, formula_text, ha='center', va='center', fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2),
            family='monospace')
    
    # 详细说明
    details = [
        '• Q, K, V: 查询、键、值向量',
        '• d_k: 每个头的维度',
        '• Softmax: 获得注意力权重',
        '• Mask: 用于decoder的因果掩码',
    ]
    for i, detail in enumerate(details):
        ax.text(7.5, 5.5 - i*0.5, detail, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('attention_mechanism_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 注意力机制流程图已保存: attention_mechanism_detailed.png")
    plt.close()


def draw_multihead_attention():
    """绘制多头注意力机制"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, 'Multi-Head Attention 结构', 
            fontsize=18, weight='bold', ha='center')
    
    # 输入
    ax.text(5, 8.8, 'Input: (batch, seq_len, d_model)', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=1.5))
    
    # 线性投影
    ax.text(2, 8, 'Linear(Q)', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    ax.text(5, 8, 'Linear(K)', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    ax.text(8, 8, 'Linear(V)', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    for x in [2, 5, 8]:
        ax.annotate('', xy=(x, 7.8), xytext=(5, 8.6),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # 分割成多个头
    ax.text(5, 7.3, '分割成 num_heads 个头', ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linestyle='dashed'))
    
    # 多个注意力头
    head_y = 6.2
    heads = []
    for i in range(4):
        x = 1.5 + i * 2
        color = ['lightcyan', 'lightgreen', 'lightyellow', 'lightcoral'][i]
        box = FancyBboxPatch((x-0.4, head_y-0.3), 0.8, 0.6, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, head_y, f'Head {i}', ha='center', va='center', fontsize=9, weight='bold')
        heads.append(x)
        
        # 箭头指向
        ax.annotate('', xy=(x, 7.0), xytext=(5, 7.3),
                    arrowprops=dict(arrowstyle='->', lw=1, color='gray'))
    
    # 每个头进行注意力计算
    ax.text(5, 5.5, 'Attention in Parallel', ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', linestyle='dashed'))
    
    # 拼接
    concat_y = 4.5
    for x in heads:
        ax.annotate('', xy=(5, concat_y+0.2), xytext=(x, 5.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    concat_box = FancyBboxPatch((4, concat_y-0.3), 2, 0.6, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(5, concat_y, 'Concat', ha='center', va='center', fontsize=11, weight='bold')
    
    # 输出投影
    ax.annotate('', xy=(5, 3.2), xytext=(5, concat_y-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    output_box = FancyBboxPatch((4, 2.7), 2, 0.5, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 2.95, 'Linear(W_o)', ha='center', va='center', fontsize=11, weight='bold')
    
    # 最终输出
    ax.annotate('', xy=(5, 1.8), xytext=(5, 2.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    output_final = FancyBboxPatch((4, 1.1), 2, 0.7, 
                                  boxstyle="round,pad=0.05", 
                                  edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(output_final)
    ax.text(5, 1.45, 'Output: (batch, seq_len, d_model)', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # 说明
    details = [
        '多头注意力的优势:',
        '• 多个表达空间: 捕捉不同的特征',
        '• 并行计算: 每个头独立运行',
        '• 丰富的表达: 综合多个视角',
        f'• 参数配置: d_model=256, num_heads=4',
        f'  → 每个头维度: 256/4=64',
    ]
    for i, detail in enumerate(details):
        fontsize = 11 if i == 0 else 9
        weight = 'bold' if i == 0 else 'normal'
        ax.text(7.5, 6.5 - i*0.4, detail, fontsize=fontsize, weight=weight, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8) if i == 0 else None)
    
    plt.tight_layout()
    plt.savefig('multihead_attention_structure.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 多头注意力结构图已保存: multihead_attention_structure.png")
    plt.close()


def draw_transformer_encoder_decoder():
    """绘制完整的 Transformer 编码器-解码器架构"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 标题
    ax.text(8, 11.5, 'Transformer 完整架构: 编码器-解码器', 
            fontsize=20, weight='bold', ha='center')
    
    # ========== 左侧: 编码器 ==========
    ax.text(3, 10.8, '编码器 (Encoder)', fontsize=14, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
    
    # 编码器输入
    ax.text(3, 10, 'Source Input\n(seq_len, d_model)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='black'))
    
    # Positional Encoding
    ax.annotate('', xy=(3, 9.2), xytext=(3, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3, 9.5, '+ Positional\nEncoding', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # 编码器层堆栈
    encoder_y = 8.5
    encoder_layers = 2
    for layer in range(encoder_layers):
        y = encoder_y - layer * 2
        
        # 多头注意力
        ax.annotate('', xy=(2, y), xytext=(3, encoder_y if layer == 0 else encoder_y - (layer-1)*2 - 0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        box_mha = FancyBboxPatch((1.2, y-0.3), 1.6, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='black', facecolor='#FFE4B5', linewidth=1.5)
        ax.add_patch(box_mha)
        ax.text(2, y, 'Multi-Head\nAttention', ha='center', va='center', fontsize=8, weight='bold')
        
        # Add & Norm
        ax.annotate('', xy=(3, y), xytext=(2.8, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        box_norm = FancyBboxPatch((2.85, y-0.25), 0.3, 0.5, 
                                  boxstyle="round,pad=0.02", 
                                  edgecolor='black', facecolor='lightgreen', linewidth=1)
        ax.add_patch(box_norm)
        ax.text(3, y, '+\nNorm', ha='center', va='center', fontsize=7, weight='bold')
        
        # FFN
        ax.annotate('', xy=(4, y), xytext=(3.2, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        box_ffn = FancyBboxPatch((4.2, y-0.3), 1.2, 0.6, 
                                 boxstyle="round,pad=0.05", 
                                 edgecolor='black', facecolor='#FFE4B5', linewidth=1.5)
        ax.add_patch(box_ffn)
        ax.text(4.8, y, 'Feed\nForward', ha='center', va='center', fontsize=8, weight='bold')
        
        # Add & Norm
        ax.annotate('', xy=(5.8, y), xytext=(5.4, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        box_norm2 = FancyBboxPatch((5.75, y-0.25), 0.3, 0.5, 
                                   boxstyle="round,pad=0.02", 
                                   edgecolor='black', facecolor='lightgreen', linewidth=1)
        ax.add_patch(box_norm2)
        ax.text(5.9, y, '+\nNorm', ha='center', va='center', fontsize=7, weight='bold')
        
        ax.text(3, y-0.8, f'Encoder Layer {layer+1}', ha='center', fontsize=8, style='italic', color='gray')
    
    # 编码器输出
    encoder_out_y = encoder_y - encoder_layers * 2 + 0.5
    ax.annotate('', xy=(3, encoder_out_y), xytext=(5.9, encoder_y - (encoder_layers-1)*2),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(3, encoder_out_y - 0.5, 'Encoder Output\n(Context)', ha='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black'))
    
    # ========== 中间: 连接 ==========
    cx = 8
    ax.annotate('', xy=(cx, 8), xytext=(5.9 + 0.3, 8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.text(cx, 8.3, 'Context\nVector', ha='center', fontsize=10, weight='bold', color='red')
    
    # ========== 右侧: 解码器 ==========
    ax.text(13, 10.8, '解码器 (Decoder)', fontsize=14, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
    
    # 解码器输入
    ax.text(13, 10, 'Target Input\n(shifted right)\n(seq_len, d_model)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcyan', edgecolor='black'))
    
    # Positional Encoding
    ax.annotate('', xy=(13, 9.2), xytext=(13, 9.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(13, 9.5, '+ Positional\nEncoding', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # 解码器层堆栈
    decoder_y = 8.5
    decoder_layers = 2
    for layer in range(decoder_layers):
        y = decoder_y - layer * 2.2
        
        # 掩码自注意力
        ax.annotate('', xy=(11.5, y), xytext=(13, decoder_y if layer == 0 else decoder_y - (layer-1)*2.2 - 0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        box_mha_mask = FancyBboxPatch((10.7, y-0.3), 1.6, 0.6, 
                                      boxstyle="round,pad=0.05", 
                                      edgecolor='red', facecolor='#FFE4FF', linewidth=1.5)
        ax.add_patch(box_mha_mask)
        ax.text(11.5, y, 'Masked MHA', ha='center', va='center', fontsize=8, weight='bold')
        ax.text(11.5, y-0.5, '(causal)', ha='center', fontsize=7, style='italic', color='red')
        
        # Add & Norm
        ax.annotate('', xy=(12.5, y), xytext=(12.3, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        box_norm_d = FancyBboxPatch((12.45, y-0.25), 0.3, 0.5, 
                                    boxstyle="round,pad=0.02", 
                                    edgecolor='black', facecolor='lightgreen', linewidth=1)
        ax.add_patch(box_norm_d)
        
        # 交叉注意力 (接收编码器输出)
        ax.annotate('', xy=(13.5, y), xytext=(12.75, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        # 来自编码器的连接
        ax.annotate('', xy=(14.2, y), xytext=(cx+0.5, 7.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='red', linestyle='dashed'))
        
        box_cross = FancyBboxPatch((13.5, y-0.3), 1.4, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='red', facecolor='#FFE4FF', linewidth=1.5)
        ax.add_patch(box_cross)
        ax.text(14.2, y, 'Cross\nAttention', ha='center', va='center', fontsize=8, weight='bold')
        
        # Add & Norm
        ax.annotate('', xy=(15.2, y), xytext=(14.8, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        box_norm_d2 = FancyBboxPatch((15.15, y-0.25), 0.3, 0.5, 
                                     boxstyle="round,pad=0.02", 
                                     edgecolor='black', facecolor='lightgreen', linewidth=1)
        ax.add_patch(box_norm_d2)
        
        # FFN
        ax.annotate('', xy=(13.5, y-1.1), xytext=(13.5, y-0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        
        box_ffn_d = FancyBboxPatch((12.9, y-1.4), 1.2, 0.6, 
                                   boxstyle="round,pad=0.05", 
                                   edgecolor='black', facecolor='#FFE4B5', linewidth=1.5)
        ax.add_patch(box_ffn_d)
        ax.text(13.5, y-1.1, 'Feed\nForward', ha='center', va='center', fontsize=8, weight='bold')
        
        ax.text(13, y-1.8, f'Decoder Layer {layer+1}', ha='center', fontsize=8, style='italic', color='gray')
    
    # 解码器输出
    decoder_out_y = decoder_y - decoder_layers * 2.2 + 0.5
    ax.annotate('', xy=(13, decoder_out_y), xytext=(13, decoder_y - (decoder_layers-1)*2.2 - 0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # 线性层 + Softmax
    ax.annotate('', xy=(13, decoder_out_y - 0.8), xytext=(13, decoder_out_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    box_linear = FancyBboxPatch((12.2, decoder_out_y - 1.3), 1.6, 0.5, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='black', facecolor='#FFE4B5', linewidth=2)
    ax.add_patch(box_linear)
    ax.text(13, decoder_out_y - 1.05, 'Linear + Softmax', ha='center', va='center', fontsize=9, weight='bold')
    
    # 最终输出
    ax.annotate('', xy=(13, decoder_out_y - 2.2), xytext=(13, decoder_out_y - 1.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.text(13, decoder_out_y - 2.7, 'Output Probabilities\n(vocab_size)', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # 图例和说明
    legend_y = 1.5
    ax.text(0.5, legend_y, '图例说明:', fontsize=11, weight='bold')
    
    items = [
        ('Attention', '#FFE4B5'),
        ('Norm', 'lightgreen'),
        ('Masked', '#FFE4FF'),
        ('I/O', 'lightcyan'),
    ]
    
    for i, (label, color) in enumerate(items):
        x = 0.5 + (i % 2) * 3
        y = legend_y - 0.5 - (i // 2) * 0.4
        rect = mpatches.Rectangle((x, y-0.15), 0.3, 0.3, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.4, y, label, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig('transformer_architecture_full.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 完整 Transformer 架构图已保存: transformer_architecture_full.png")
    plt.close()


def draw_data_flow():
    """绘制数据流向图"""
    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # 标题
    ax.text(7, 10.5, 'Transformer 数据流向图', 
            fontsize=18, weight='bold', ha='center')
    
    # Source Input
    y_pos = 9.5
    ax.text(2, y_pos, 'Source Text\n"Hello World"', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
    
    # Tokenization
    ax.annotate('', xy=(2, y_pos-0.8), xytext=(2, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.5, y_pos-0.55, 'Tokenization', fontsize=9, style='italic')
    
    # Embedding
    y_pos -= 1.2
    ax.text(2, y_pos, 'Token IDs\n[5, 12, 9, ...]', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    ax.annotate('', xy=(2, y_pos-0.8), xytext=(2, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.5, y_pos-0.55, 'Embedding\n(d_model=256)', fontsize=9, style='italic')
    
    # Positional Encoding
    y_pos -= 1.2
    ax.text(2, y_pos, 'Embeddings\n(seq_len, 256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    ax.annotate('', xy=(2, y_pos-0.8), xytext=(2, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.5, y_pos-0.55, '+ Positional\nEncoding', fontsize=9, style='italic')
    
    # Encoder
    y_pos -= 1.2
    ax.text(2, y_pos, 'Encoder Input\n(seq_len, 256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    ax.annotate('', xy=(2, y_pos-0.8), xytext=(2, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(3.5, y_pos-0.55, 'Pass through\n2 Encoder Layers', fontsize=9, style='italic')
    
    # Encoder Output
    y_pos -= 1.2
    ax.text(2, y_pos, 'Context Vector\n(seq_len, 256)', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # ==================== 右侧: 解码器流程 ====================
    
    # Target Input
    y_pos = 9.5
    ax.text(12, y_pos, 'Target Text\n"<START> ..."', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, 'Tokenization', fontsize=9, style='italic')
    
    # Target Embedding
    y_pos -= 1.2
    ax.text(12, y_pos, 'Token IDs\n[1, ...]', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, 'Embedding\n(d_model=256)', fontsize=9, style='italic')
    
    # Positional Encoding
    y_pos -= 1.2
    ax.text(12, y_pos, 'Embeddings\n(seq_len, 256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, '+ Positional\nEncoding', fontsize=9, style='italic')
    
    # Decoder with Context
    y_pos -= 1.2
    ax.text(12, y_pos, 'Decoder Input\n(seq_len, 256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    # 从 Encoder 获取 Context
    ax.annotate('', xy=(10, y_pos), xytext=(3.5, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='red', linestyle='dashed'))
    ax.text(6.5, 5.2, 'Context Vector (from Encoder)', fontsize=9, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red', linestyle='dashed'))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, 'Pass through\n2 Decoder Layers\n(with context)', fontsize=9, style='italic')
    
    # Decoder Output
    y_pos -= 1.2
    ax.text(12, y_pos, 'Decoder Output\n(seq_len, 256)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, 'Linear Projection\nto Vocab', fontsize=9, style='italic')
    
    # Final Output
    y_pos -= 1.2
    ax.text(12, y_pos, 'Logits\n(seq_len, vocab_size)', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black'))
    
    ax.annotate('', xy=(12, y_pos-0.8), xytext=(12, y_pos-0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(10.5, y_pos-0.55, 'Softmax &\nGreedy Decoding', fontsize=9, style='italic')
    
    # Output
    y_pos -= 1.2
    ax.text(12, y_pos, 'Output Text\n"Hello ..."', ha='center', fontsize=10, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # 中间说明框
    box_text = """数据形状变化:
① Source: 字符串 → Token IDs
② Token IDs: (seq_len,) → 整数
③ Embeddings: (seq_len, vocab_size) → 词嵌入
④ Transformer: (seq_len, d_model) 保持
⑤ Output: (seq_len, vocab_size) → 概率"""
    
    ax.text(7, 3, box_text, ha='center', va='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2, pad=0.5))
    
    plt.tight_layout()
    plt.savefig('transformer_data_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 数据流向图已保存: transformer_data_flow.png")
    plt.close()


def draw_training_process():
    """绘制训练过程流程图"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(7, 9.5, 'Transformer 训练过程流程', 
            fontsize=18, weight='bold', ha='center')
    
    # 步骤 1: 数据
    y = 8.5
    ax.text(7, y, '1. 加载训练数据', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(7, y-0.4, '100 个序列对，词表大小 20', fontsize=9, style='italic')
    
    # 步骤 2: 初始化
    y -= 1.2
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(7, y, '2. 模型初始化', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(7, y-0.4, '参数数: 2.65M，优化器: Adam', fontsize=9, style='italic')
    
    # 循环开始
    y -= 1.2
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    loop_box = FancyBboxPatch((3.5, y-0.5), 7, 3.5, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='red', facecolor='white', linewidth=2, linestyle='dashed')
    ax.add_patch(loop_box)
    ax.text(10.5, y+2.8, '对每个 Epoch 循环', fontsize=10, weight='bold', color='red',
            style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 步骤 3: 前向传播
    y -= 0.7
    ax.text(7, y, '3. 前向传播', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black', linewidth=1.5))
    ax.text(7, y-0.4, 'Input → Encoder → Decoder → Output', fontsize=9, style='italic')
    
    # 步骤 4: 计算损失
    y -= 1.2
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(7, y, '4. 计算损失函数', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black', linewidth=1.5))
    ax.text(7, y-0.4, 'Loss = CrossEntropyLoss(output, target)', fontsize=9, style='italic')
    
    # 步骤 5: 反向传播
    y -= 1.2
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(7, y, '5. 反向传播 (Backprop)', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black', linewidth=1.5))
    ax.text(7, y-0.4, '计算梯度: dL/dθ', fontsize=9, style='italic')
    
    # 步骤 6: 参数更新
    y -= 1.2
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax.text(7, y, '6. 参数更新', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFE4B5', edgecolor='black', linewidth=1.5))
    ax.text(7, y-0.4, 'θ = θ - lr × dL/dθ', fontsize=9, style='italic')
    
    # 循环箭头
    ax.annotate('', xy=(2.5, 5.2), xytext=(6.8, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
    ax.text(2, 4, 'Next Batch\n(20 epochs)', fontsize=9, weight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red', linestyle='dashed'))
    
    # 步骤 7: 验证
    y -= 2
    ax.annotate('', xy=(7, y+0.5), xytext=(7, 3),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(7, y, '7. 验证模型', fontsize=11, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(7, y-0.4, '测试准确率: 100% ✓', fontsize=9, style='italic')
    
    # 最终结果
    y -= 1
    ax.annotate('', xy=(7, y+0.3), xytext=(7, y+0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(7, y, '✓ 训练完成！', fontsize=12, weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', edgecolor='black', linewidth=2))
    
    # 左侧说明
    details = [
        '训练统计:',
        '• Epochs: 20',
        '• Batch size: 32',
        '• Learning rate: 0.0005',
        '• Loss: CrossEntropyLoss',
        '',
        '结果:',
        '• 初始 Loss: 1.6312',
        '• 最终 Loss: 0.0008',
        '• 准确率: 100%',
    ]
    
    for i, detail in enumerate(details):
        fontsize = 10 if detail.startswith('训') or detail.startswith('结') else 9
        weight = 'bold' if detail.startswith('训') or detail.startswith('结') else 'normal'
        color = 'black' if detail else 'white'
        ax.text(0.5, 8.5 - i*0.35, detail, fontsize=fontsize, weight=weight, color=color, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray') if detail and not detail.startswith('•') else None)
    
    plt.tight_layout()
    plt.savefig('transformer_training_process.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 训练过程流程图已保存: transformer_training_process.png")
    plt.close()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("生成 Transformer 架构流程图...")
    print("="*60 + "\n")
    
    draw_attention_mechanism()
    draw_multihead_attention()
    draw_transformer_encoder_decoder()
    draw_data_flow()
    draw_training_process()
    
    print("\n" + "="*60)
    print("✅ 所有架构流程图生成完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  1. attention_mechanism_detailed.png")
    print("  2. multihead_attention_structure.png")
    print("  3. transformer_architecture_full.png")
    print("  4. transformer_data_flow.png")
    print("  5. transformer_training_process.png")
