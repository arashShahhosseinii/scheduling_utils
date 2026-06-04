from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.distributions import Categorical


class MaskedCategoricalDistribution:
    """Simple distribution with action masking support for PPO."""

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


class GraphStateMLPExtractor(BaseFeaturesExtractor):
    """
    Fast feature extractor for graph observations.
    Encodes node features with an MLP and then does a masked mean pool.
    Encodes core and global features with separate MLPs.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        node_dim = int(observation_space["node_features"].shape[1])
        core_shape = observation_space["core_features"].shape
        global_dim = int(observation_space["global_features"].shape[0])

        self.num_cores = int(core_shape[0])
        self.core_dim = int(core_shape[1])

        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Core feature encoder
        self.core_encoder = nn.Sequential(
            nn.Linear(self.num_cores * self.core_dim, 64),
            nn.ReLU(),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 64),
            nn.ReLU(),
        )

        # Final fusion MLP
        self.out = nn.Sequential(
            nn.Linear(128 + 64 + 64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = obs["node_features"].float()          # (B, N, node_dim)
        node_mask = obs["node_mask"].bool()       # (B, N)

        # Encode nodes
        h = F.relu(self.node_encoder(x))          # (B, N, 128)

        # Masked mean pool over nodes
        mask = node_mask.unsqueeze(-1).float()
        graph_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # Encode cores and global
        core_emb = self.core_encoder(obs["core_features"].float().flatten(start_dim=1))
        global_emb = self.global_encoder(obs["global_features"].float())

        # Concatenate and output
        return self.out(torch.cat([graph_emb, core_emb, global_emb], dim=1))


class MaskedMLPActorCriticPolicy(ActorCriticPolicy):
    """
    PPO policy that uses the fast MLP feature extractor and action masking.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("features_extractor_class", GraphStateMLPExtractor)
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
        self, obs: Dict[str, torch.Tensor]
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