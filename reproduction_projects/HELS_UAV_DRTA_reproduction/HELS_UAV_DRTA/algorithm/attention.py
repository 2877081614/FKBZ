"""
注意力机制状态编码器 (论文 Eq.43-45, Fig.8)
Query = HELS自身状态, Key = UAV状态, Value = UAV状态
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionEncoder(nn.Module):
    """将可变长度UAV状态编码为固定维度特征向量"""

    def __init__(self, hels_dim=6, uav_dim=5, d_k=64, d_v=64, d_attn=128, env_dim=1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_attn = d_attn
        self.scale = math.sqrt(d_k)

        # Eq.(43): Linear projections
        self.W_q = nn.Linear(hels_dim + env_dim, d_k)
        self.W_k = nn.Linear(uav_dim, d_k)
        self.W_v = nn.Linear(uav_dim, d_v)

        # Output projection
        self.out_proj = nn.Linear(d_v, d_attn)

    def forward(self, hels_state, uav_states, env_param, mask=None):
        """
        Args:
            hels_state:  (B, hels_dim) HELS自身状态
            uav_states:  (B, N, uav_dim) 所有UAV状态
            env_param:   (B, env_dim) 大气环境参数
            mask:        (B, N) 有效UAV mask (1=有效)
        Returns:
            encoded:      (B, d_attn)
            attn_weights: (B, N)
        """
        B = hels_state.shape[0]
        N = uav_states.shape[1]

        # Eq.(43): Q = W_q([s_LaWS, env]), K = W_k(s_UAV), V = W_v(s_UAV)
        Q = self.W_q(torch.cat([hels_state, env_param], dim=-1))  # (B, d_k)
        K = self.W_k(uav_states)  # (B, N, d_k)
        V = self.W_v(uav_states)  # (B, N, d_v)

        # Eq.(44): a_ij = softmax(Q·K^T / sqrt(d_k))
        scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2))  # (B, 1, N)
        scores = scores / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, N)

        # Eq.(45): encoded = sum_j a_ij * V_j
        context = torch.bmm(attn_weights, V).squeeze(1)  # (B, d_v)
        encoded = self.out_proj(context)  # (B, d_attn)

        return encoded, attn_weights.squeeze(1)
