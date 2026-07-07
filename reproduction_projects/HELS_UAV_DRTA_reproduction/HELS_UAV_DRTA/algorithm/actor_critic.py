"""
Actor-Critic网络 (Section 3.1)
"""
import torch
import torch.nn as nn


class Actor(nn.Module):
    """策略网络 — 每个HELS独立, 从attention编码→动作logits"""

    def __init__(self, attn_dim=128, n_actions=51, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, attn_encoded):
        return self.net(attn_encoded)  # (B, n_actions)


class Critic(nn.Module):
    """集中式Critic — 使用全局信息(所有agent的attention编码+联合动作)"""

    def __init__(self, n_agents, attn_dim=128, n_actions=51, hidden_dim=512):
        super().__init__()
        input_dim = n_agents * attn_dim + n_agents * n_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_encodings, joint_actions_onehot):
        x = torch.cat([global_encodings, joint_actions_onehot], dim=-1)
        return self.net(x)  # (B, 1)
