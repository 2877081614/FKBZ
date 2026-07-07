"""
RND内在奖励模块 (论文 Eq.46-47, Fig.9)
基于Random Network Distillation的探索奖励
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNDNetwork(nn.Module):
    """RND中的MLP网络"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RNDIntrinsicReward(nn.Module):
    """RND内在奖励: 预测误差越大 → 状态越新颖 → 奖励越大"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64,
                 beta_r0=0.1, k_r=1e-4, lr=1e-4):
        super().__init__()
        # Target network phi: fixed random, never trained
        self.target = RNDNetwork(input_dim, hidden_dim, output_dim)
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network phi_tilde: trained to match target
        self.predictor = RNDNetwork(input_dim, hidden_dim, output_dim)

        self.beta_r0 = beta_r0
        self.k_r = k_r
        self.training_step = 0
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic_reward(self, state):
        """
        Eq.(46): r_c = beta_r * ||phi_tilde(state) - phi(state)||^2
        Returns scalar reward per sample (batch,)
        """
        with torch.no_grad():
            target_feat = self.target(state)
        pred_feat = self.predictor(state)
        mse = ((pred_feat - target_feat) ** 2).mean(dim=-1)  # (B,)
        beta_r = self.beta_r0 * np.exp(-self.k_r * self.training_step)
        return beta_r * mse

    def update(self, state):
        """Train predictor to match target output"""
        with torch.no_grad():
            target_feat = self.target(state)
        pred_feat = self.predictor(state)
        loss = F.mse_loss(pred_feat, target_feat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1
        return loss.item()

    @staticmethod
    def hybrid_reward(extrinsic, intrinsic, gamma_mix=0.9):
        """Eq.(47): r_h = gamma * r_e + (1-gamma) * r_c"""
        return gamma_mix * extrinsic + (1 - gamma_mix) * intrinsic
