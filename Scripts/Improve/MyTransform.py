import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 把每个 token 一次性映射成 q、k、v 三组向量
        self.qkv = nn.Linear(dim, dim * 3)
        # 对注意力聚合后的结果再做一次线性变换
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        if self.verbose:
            print("[SelfAttention] 进入 forward")

        # 1. 输入序列: x 的形状是 (B, N, D)
        # 作用: 把一批 token 序列送入注意力模块
        B, N, D = x.shape
        if self.verbose:
            print(f"[SelfAttention] 输入形状: {x.shape}")

        # 2. 生成 QKV
        # 作用: 为每个 token 准备
        # q(query): 我需要从别人那里找什么信息
        # k(key): 我能提供什么信息给别人匹配
        # v(value): 真正被聚合和传递的内容
        if self.verbose:
            print("[SelfAttention] 生成 QKV")
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 3. 算注意力: Q 和 K 先算相关性，再对 V 做加权求和
        # 作用: 让每个 token 根据和其他 token 的相关程度, 聚合全局信息
        if self.verbose:
            print("[SelfAttention] 计算注意力分数和加权结果")
        scores = q @ k.transpose(-2, -1) / (D**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        # 再做一次线性映射, 整理注意力输出的表达空间
        out = self.proj(out)
        if self.verbose:
            print(f"[SelfAttention] 输出形状: {out.shape}")
            print("[SelfAttention] 离开 forward")
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, verbose=False):
        super().__init__()
        self.verbose = verbose
        # 先归一化再做注意力, 让训练更稳定
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, verbose=verbose)
        # 第二次归一化, 给前馈网络准备更稳定的输入
        self.norm2 = nn.LayerNorm(dim)
        # 前馈网络: 不做 token 之间交互, 只增强每个 token 自己的特征表达
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        if self.verbose:
            print("[TransformerBlock] 进入 forward")
            print(f"[TransformerBlock] 输入形状: {x.shape}")

        # 4. 残差: 输入 x 先经过注意力分支，再和原始 x 相加
        # 作用: 保留原始信息, 防止注意力分支把原特征完全改坏, 也更利于训练
        if self.verbose:
            print("[TransformerBlock] 调用 norm1")
        x_norm = self.norm1(x)
        if self.verbose:
            print("[TransformerBlock] 调用 SelfAttention")
        attn_out = self.attn(x_norm)
        if self.verbose:
            print("[TransformerBlock] 执行第一次残差相加")
        x = x + attn_out

        # 5. 前馈网络: 每个 token 独立通过两层 MLP
        # 作用: 在注意力完成“信息交换”后, 进一步提炼每个 token 自身的非线性表达
        # 6. 残差: MLP 分支输出再和上一层结果相加
        # 作用: 保留注意力后的主干信息, 让 MLP 只负责增量修正
        if self.verbose:
            print("[TransformerBlock] 调用 norm2")
        x_norm = self.norm2(x)
        if self.verbose:
            print("[TransformerBlock] 调用 MLP")
        mlp_out = self.mlp(x_norm)
        if self.verbose:
            print("[TransformerBlock] 执行第二次残差相加")
        x = x + mlp_out
        if self.verbose:
            print(f"[TransformerBlock] 输出形状: {x.shape}")
            print("[TransformerBlock] 离开 forward")
        return x


# 随机生成一个输入张量，形状是 2 × 16 × 64
x = torch.randn(2, 16, 64)
# 输入和输出的 token 维度都是 64
# 前馈网络中间层先扩展到 128，再投影回 64
block = TransformerBlock(64, 128, verbose=True)
y = block(x)
print(y.shape)
