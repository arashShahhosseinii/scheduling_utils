from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


class SimpleGATLayer(nn.Module):
    """Small dependency-free multi-head GAT layer for batched padded DAGs."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, N, F]
        # edge_index: [B, 2, E]
        bsz, num_nodes, _ = x.shape

        h = self.lin(x).view(bsz, num_nodes, self.num_heads, self.head_dim)
        out = torch.zeros_like(h)

        src_all = edge_index[:, 0, :].long()
        dst_all = edge_index[:, 1, :].long()
        edge_valid_all = edge_mask.bool()

        # اضافه کردن self-loop برای اینکه هر node اطلاعات خودش را هم حفظ کند
        loop_nodes = torch.arange(num_nodes, device=x.device).view(1, -1).expand(bsz, -1)
        loop_valid = node_mask.bool()

        src_all = torch.cat([src_all, loop_nodes], dim=1)
        dst_all = torch.cat([dst_all, loop_nodes], dim=1)
        edge_valid_all = torch.cat([edge_valid_all, loop_valid], dim=1)

        for b in range(bsz):
            src = src_all[b][edge_valid_all[b]]
            dst = dst_all[b][edge_valid_all[b]]

            if src.numel() == 0:
                continue

            h_src = h[b, src]
            h_dst = h[b, dst]

            score = (h_src * self.attn_src).sum(-1) + (h_dst * self.attn_dst).sum(-1)
            score = F.leaky_relu(score, negative_slope=0.2)

            # softmax روی edgeهای ورودی هر destination node
            for head in range(self.num_heads):
                head_score = score[:, head]

                for node in torch.unique(dst):
                    idx = dst == node
                    alpha = torch.softmax(head_score[idx], dim=0)
                    alpha = self.dropout(alpha)

                    msg = (alpha.unsqueeze(-1) * h_src[idx, head, :]).sum(dim=0)
                    out[b, node, head, :] = msg

        out = out.reshape(bsz, num_nodes, self.num_heads * self.head_dim) + self.bias
        out = out * node_mask.unsqueeze(-1).float()
        return out


class GraphStateGATExtractor(BaseFeaturesExtractor):
    """
    این کلاس state گرافی محیط را می‌گیرد و با GAT به یک latent vector تبدیل می‌کند.
    PPO بعد از این latent vector تصمیم‌گیری می‌کند.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        gat_hidden_dim: int = 128,
        gat_heads: int = 4,
        gat_layers: int = 2,
        features_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim)

        node_dim = int(observation_space["node_features"].shape[1])
        core_shape = observation_space["core_features"].shape
        global_dim = int(observation_space["global_features"].shape[0])

        self.num_cores = int(core_shape[0])
        self.core_dim = int(core_shape[1])

        self.input_proj = nn.Linear(node_dim, gat_hidden_dim)

        self.gat_layers = nn.ModuleList(
            [
                SimpleGATLayer(
                    in_dim=gat_hidden_dim,
                    out_dim=gat_hidden_dim,
                    num_heads=gat_heads,
                    dropout=dropout,
                )
                for _ in range(gat_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(gat_hidden_dim) for _ in range(gat_layers)]
        )

        self.core_encoder = nn.Sequential(
            nn.Linear(self.num_cores * self.core_dim, 64),
            nn.ReLU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Linear(gat_hidden_dim + 64 + 64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs["node_features"].float()
        edge_index = obs["edge_index"].long()
        edge_mask = obs["edge_mask"].bool()
        node_mask = obs["node_mask"].bool()

        h = F.relu(self.input_proj(x))

        for gat, norm in zip(self.gat_layers, self.norms):
            h2 = F.elu(gat(h, edge_index, edge_mask, node_mask))
            h = norm(h + h2)

        mask = node_mask.unsqueeze(-1).float()
        graph_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        core_emb = self.core_encoder(obs["core_features"].float().flatten(start_dim=1))
        global_emb = self.global_encoder(obs["global_features"].float())

        return self.out(torch.cat([graph_emb, core_emb, global_emb], dim=1))


class MaskedCategoricalDistribution:
    """Distribution ساده برای پشتیبانی از action_mask در PPO."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.distribution = Categorical(logits=logits)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return torch.argmax(self.distribution.logits, dim=1)
        return self.distribution.sample()

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        return self.distribution.log_prob(actions.long())

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.logits, dim=1)


class MaskedGATActorCriticPolicy(ActorCriticPolicy):
    """
    Policy مخصوص Stable-Baselines3 PPO.

    این policy دو کار انجام می‌دهد:
    1. state گرافی را با GAT encode می‌کند.
    2. فقط actionهای معتبر را برای PPO فعال نگه می‌دارد.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("features_extractor_class", GraphStateGATExtractor)
        kwargs.setdefault("net_arch", dict(pi=[128, 128], vf=[128, 128]))
        super().__init__(*args, **kwargs)

    @staticmethod
    def _apply_action_mask(logits: torch.Tensor, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = obs.get("action_mask")

        if mask is None:
            return logits

        mask = mask.bool()

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        has_valid = mask.any(dim=1, keepdim=True)
        safe_mask = torch.where(has_valid, mask, torch.ones_like(mask))

        return logits.masked_fill(~safe_mask, torch.finfo(logits.dtype).min)

    def _masked_distribution(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[MaskedCategoricalDistribution, torch.Tensor]:
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            features = features[0]

        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        logits = self._apply_action_mask(logits, obs)

        values = self.value_net(latent_vf)

        return MaskedCategoricalDistribution(logits), values

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        dist, values = self._masked_distribution(obs)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        dist, values = self._masked_distribution(obs)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: Dict[str, torch.Tensor]) -> MaskedCategoricalDistribution:
        dist, _ = self._masked_distribution(obs)
        return dist

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.extract_features(obs)

        if isinstance(features, tuple):
            features = features[0]

        _, latent_vf = self.mlp_extractor(features)
        return self.value_net(latent_vf)