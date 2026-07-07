"""
MADDPG-IA主算法 (Algorithm 1 伪代码)
I = Intrinsic Reward (RND), A = Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from algorithm.attention import AttentionEncoder
from algorithm.rnd import RNDIntrinsicReward
from algorithm.actor_critic import Actor, Critic
from algorithm.replay_buffer import ReplayBuffer
from algorithm.gumbel_softmax import gumbel_softmax


class MADDPG_IA:
    """MADDPG with Intrinsic reward & Attention"""

    def __init__(self, n_agents, n_actions, n_uavs,
                 hels_dim=6, uav_dim=5, env_dim=1,
                 d_attn=128, d_k=64, actor_hidden=256, critic_hidden=512,
                 actor_lr=1e-3, critic_lr=1e-3, rnd_lr=1e-4,
                 gamma=0.95, tau=0.01, batch_size=2048, buffer_size=int(1e6),
                 gumbel_temp=1.0, gumbel_min=0.1, gumbel_anneal=0.9995,
                 device='cuda'):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_uavs = n_uavs
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device

        # --- Shared Attention Encoder ---
        self.attention = AttentionEncoder(
            hels_dim=hels_dim, uav_dim=uav_dim,
            d_k=d_k, d_v=d_k, d_attn=d_attn, env_dim=env_dim
        ).to(device)

        # --- Actor networks (one per agent) ---
        self.actors = nn.ModuleList([
            Actor(attn_dim=d_attn, n_actions=n_actions, hidden_dim=actor_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self.target_actors = nn.ModuleList([
            Actor(attn_dim=d_attn, n_actions=n_actions, hidden_dim=actor_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self._hard_update(self.actors, self.target_actors)

        # --- Critic networks (one centralized critic per agent) ---
        self.critics = nn.ModuleList([
            Critic(n_agents=n_agents, attn_dim=d_attn,
                   n_actions=n_actions, hidden_dim=critic_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self.target_critics = nn.ModuleList([
            Critic(n_agents=n_agents, attn_dim=d_attn,
                   n_actions=n_actions, hidden_dim=critic_hidden)
            for _ in range(n_agents)
        ]).to(device)
        self._hard_update(self.critics, self.target_critics)

        # --- Optimizers ---
        self.actor_optims = [optim.Adam(self.actors[i].parameters(), lr=actor_lr)
                             for i in range(n_agents)]
        self.critic_optims = [optim.Adam(self.critics[i].parameters(), lr=critic_lr)
                              for i in range(n_agents)]

        # --- RND modules (one per agent) ---
        self.rnds = nn.ModuleList([
            RNDIntrinsicReward(input_dim=d_attn, lr=rnd_lr)
            for _ in range(n_agents)
        ]).to(device)

        # --- Replay Buffer ---
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # --- Gumbel-Softmax temperature ---
        self.gumbel_temp = gumbel_temp
        self.gumbel_min = gumbel_min
        self.gumbel_anneal = gumbel_anneal

    def _hard_update(self, src, tgt):
        for s, t in zip(src.parameters(), tgt.parameters()):
            t.data.copy_(s.data)

    def _soft_update(self, src, tgt):
        """Eq.(37): theta' = tau * theta + (1-tau) * theta'"""
        for s, t in zip(src.parameters(), tgt.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    # ------------------------------------------------------------------
    # Encode observation using attention
    # ------------------------------------------------------------------
    def encode_obs(self, obs_i, batch=False):
        """Encode one agent's observation via attention encoder"""
        if not batch:
            hels_s = torch.FloatTensor(obs_i[:6]).unsqueeze(0).to(self.device)
            env_p = torch.FloatTensor([obs_i[6]]).unsqueeze(0).to(self.device)
            uav_flat = obs_i[7:]
            uav_s = torch.FloatTensor(uav_flat).view(1, self.n_uavs, 5).to(self.device)
        else:
            # obs_i shape: (B, obs_dim)
            hels_s = obs_i[:, :6].to(self.device)
            env_p = obs_i[:, 6:7].to(self.device)
            uav_s = obs_i[:, 7:].view(obs_i.shape[0], self.n_uavs, 5).to(self.device)

        mask = (uav_s.abs().sum(dim=-1) > 1e-6).float()
        encoded, attn_w = self.attention(hels_s, uav_s, env_p, mask)
        return encoded, attn_w

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs_i, evaluate=False):
        """Select action for agent i"""
        encoded, _ = self.encode_obs(obs_i)
        with torch.no_grad():
            logits = self.actors[0](encoded)  # shared actor → use agent 0

        if evaluate or self.gumbel_temp <= self.gumbel_min:
            return torch.argmax(logits, dim=-1).item(), encoded
        else:
            onehot = gumbel_softmax(logits, self.gumbel_temp, hard=True)
            return torch.argmax(onehot, dim=-1).item(), encoded

    # ------------------------------------------------------------------
    # Update step (two-pass: critics first with detached encodings, then actors)
    # ------------------------------------------------------------------
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        device = self.device

        def _stack_obs(key):
            return torch.FloatTensor(np.stack([b['obs'][key] for b in batch])).to(device)
        def _stack_next_obs(key):
            return torch.FloatTensor(np.stack([b['next_obs'][key] for b in batch])).to(device)
        def _stack_rew(key):
            return torch.FloatTensor([b['rewards'][key] for b in batch]).unsqueeze(-1).to(device)
        def _stack_act(key):
            return torch.LongTensor([b['actions'][key] for b in batch]).to(device)
        dones = torch.FloatTensor([1.0 if b['dones'] else 0.0 for b in batch]).unsqueeze(-1).to(device)

        # Encode all states (will be recomputed for actor pass to get fresh graphs)
        curr_enc_det = []
        next_enc_det = []
        curr_act_1hot = []
        for i in range(self.n_agents):
            obs_i = _stack_obs(f'agent_{i}')
            nobs_i = _stack_next_obs(f'agent_{i}')
            with torch.no_grad():
                ce, _ = self.encode_obs(obs_i, batch=True)
                ne, _ = self.encode_obs(nobs_i, batch=True)
            curr_enc_det.append(ce.detach())
            next_enc_det.append(ne.detach())
            act_i = _stack_act(f'agent_{i}')
            a1h = F.one_hot(act_i, num_classes=self.n_actions).float()
            curr_act_1hot.append(a1h)

        global_enc_det = torch.cat(curr_enc_det, dim=-1)
        next_global_enc_det = torch.cat(next_enc_det, dim=-1)
        joint_act = torch.cat(curr_act_1hot, dim=-1)

        critic_losses, actor_losses, rnd_losses = [], [], []

        # ---- Pass 1: Update all Critics (detached encodings) ----
        for i in range(self.n_agents):
            with torch.no_grad():
                next_act_1hot = []
                for j in range(self.n_agents):
                    nlogits = self.target_actors[j](next_enc_det[j])
                    na1h = gumbel_softmax(nlogits, temperature=0.1, hard=True)
                    next_act_1hot.append(na1h)
                next_joint_act = torch.cat(next_act_1hot, dim=-1)
                target_q = self.target_critics[i](next_global_enc_det, next_joint_act)
                rew_i = _stack_rew(f'agent_{i}')
                td_target = rew_i + self.gamma * (1 - dones) * target_q

            current_q = self.critics[i](global_enc_det, joint_act)
            critic_loss = F.mse_loss(current_q, td_target)
            self.critic_optims[i].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0)
            self.critic_optims[i].step()
            critic_losses.append(critic_loss.item())

        # ---- Pass 2: Update all Actors (fresh encodings with gradients) ----
        curr_enc = []
        for i in range(self.n_agents):
            obs_i = _stack_obs(f'agent_{i}')
            ce, _ = self.encode_obs(obs_i, batch=True)
            curr_enc.append(ce)
        global_enc = torch.cat(curr_enc, dim=-1)

        for i in range(self.n_agents):
            new_logits = self.actors[i](curr_enc[i])
            new_act = gumbel_softmax(new_logits, self.gumbel_temp)
            joint_new = []
            for j in range(self.n_agents):
                if j == i:
                    joint_new.append(new_act)
                else:
                    joint_new.append(curr_act_1hot[j].detach())
            joint_new_cat = torch.cat(joint_new, dim=-1)
            q_val = self.critics[i](global_enc.detach(), joint_new_cat)
            actor_loss = -q_val.mean()
            self.actor_optims[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actor_optims[i].step()
            actor_losses.append(actor_loss.item())

        # ---- Pass 3: Soft update targets (Eq.37) ----
        for i in range(self.n_agents):
            self._soft_update(self.actors[i], self.target_actors[i])
            self._soft_update(self.critics[i], self.target_critics[i])

        # ---- Pass 4: Update RND ----
        for i in range(self.n_agents):
            nobs_i = _stack_next_obs(f'agent_{i}')
            with torch.no_grad():
                ne, _ = self.encode_obs(nobs_i, batch=True)
            rnd_loss = self.rnds[i].update(ne.detach())
            rnd_losses.append(rnd_loss)

        # --- Anneal Gumbel temperature ---
        self.gumbel_temp = max(self.gumbel_min, self.gumbel_temp * self.gumbel_anneal)

        return {
            'critic_loss': np.mean(critic_losses),
            'actor_loss': np.mean(actor_losses),
            'rnd_loss': np.mean(rnd_losses),
            'gumbel_temp': self.gumbel_temp,
        }

    # ------------------------------------------------------------------
    # Intrinsic reward computation for env interaction
    # ------------------------------------------------------------------
    def compute_mixed_reward(self, next_obs_i, env_reward):
        """Compute hybrid reward (Eq.47)"""
        encoded, _ = self.encode_obs(next_obs_i)
        r_c = self.rnds[0].compute_intrinsic_reward(encoded)  # use agent 0's RND
        r_h = self.rnds[0].hybrid_reward(env_reward, r_c.item())
        return r_h

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'attention': self.attention.state_dict(),
            'actors': {i: self.actors[i].state_dict() for i in range(self.n_agents)},
            'target_actors': {i: self.target_actors[i].state_dict() for i in range(self.n_agents)},
            'critics': {i: self.critics[i].state_dict() for i in range(self.n_agents)},
            'target_critics': {i: self.target_critics[i].state_dict() for i in range(self.n_agents)},
            'rnds': {i: self.rnds[i].state_dict() for i in range(self.n_agents)},
            'gumbel_temp': self.gumbel_temp,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.attention.load_state_dict(ckpt['attention'])
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt['actors'][i])
            self.target_actors[i].load_state_dict(ckpt['target_actors'][i])
            self.critics[i].load_state_dict(ckpt['critics'][i])
            self.target_critics[i].load_state_dict(ckpt['target_critics'][i])
            self.rnds[i].load_state_dict(ckpt['rnds'][i])
        self.gumbel_temp = ckpt['gumbel_temp']
