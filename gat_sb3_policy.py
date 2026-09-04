from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch_geometric.nn import GATConv


class _UnusedDictExtractor(BaseFeaturesExtractor):
    """SB3 constructor shim; graph processing is done by the custom network."""

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return observations["global_features"].float()[:, :1]


class DirectedGATBlock(nn.Module):
    """Bidirectional GAT block over predecessor and successor directions."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        attention_dropout: float,
        feature_dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or heads <= 0 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by heads.")

        head_dim = hidden_dim // heads
        self.predecessor_attention = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=True,
            residual=False,
        )
        self.successor_attention = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=heads,
            concat=True,
            dropout=attention_dropout,
            add_self_loops=True,
            residual=False,
        )
        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(feature_dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        predecessor_messages = self.predecessor_attention(
            node_embeddings,
            edge_index,
        )
        successor_messages = self.successor_attention(
            node_embeddings,
            edge_index.flip(0),
        )
        messages = torch.cat(
            [predecessor_messages, successor_messages],
            dim=-1,
        )
        messages = F.elu(self.fusion(messages))
        messages = self.dropout(messages)
        output = self.norm(node_embeddings + messages)
        return output * node_mask.unsqueeze(-1).to(output.dtype)


class GraphAttentionGangSchedulerNetwork(nn.Module):
    """
    GAT actor-critic for Task x Gang-composition actions.

    The actor does not identify concrete physical cores. For each task and slot,
    it scores the composition implied by:

        slot      = A12 count
        A7 count  = m_i - slot

    Physical-core identities are selected later by DagSchedulingEnv.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        core_hidden_dim: int = 32,
        global_hidden_dim: int = 32,
        allocation_hidden_dim: int = 32,
        context_hidden_dim: int = 64,
        graph_pool_dim: int = 128,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
        attention_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("GraphAttentionGangSchedulerNetwork requires Dict observations.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("Gang scheduler requires a Discrete action space.")
        if gat_layers <= 0 or gat_hidden_dim % gat_heads != 0:
            raise ValueError("Invalid GAT layer/head configuration.")

        node_shape = observation_space["node_features"].shape
        core_shape = observation_space["core_features"].shape
        global_shape = observation_space["global_features"].shape

        self.max_tasks = int(node_shape[0])
        self.node_feature_dim = int(node_shape[1])
        self.num_resource_types = int(core_shape[0])
        self.core_feature_dim = int(core_shape[1])
        self.global_feature_dim = int(global_shape[0])

        if int(action_space.n) % self.max_tasks != 0:
            raise ValueError("Action count must be divisible by max_tasks.")
        self.num_action_slots = int(action_space.n) // self.max_tasks
        self.max_gang_size = self.num_action_slots - 1
        if self.max_gang_size <= 0:
            raise ValueError("Gang action space needs at least slots 0 and 1.")

        self.node_input = nn.Sequential(
            nn.Linear(self.node_feature_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(gat_hidden_dim),
        )
        self.gat_blocks = nn.ModuleList(
            [
                DirectedGATBlock(
                    hidden_dim=gat_hidden_dim,
                    heads=gat_heads,
                    attention_dropout=attention_dropout,
                    feature_dropout=dropout,
                )
                for _ in range(gat_layers)
            ]
        )

        if graph_pool_dim <= 0:
            raise ValueError("graph_pool_dim must be positive.")
        self.graph_pool_dim = int(graph_pool_dim)

        # Learnable graph pooling requested by the project supervisor:
        #   1) masked mean over node embeddings
        #   2) masked max over node embeddings
        #   3) concatenate [mean || max]
        #   4) trainable projection to graph_pool_dim (default: 128)
        #
        # With gat_hidden_dim=64, the concatenated statistics have size 128.
        # The projection is still kept trainable so the network can learn how
        # to mix mean- and max-based graph information before PPO consumes it.
        self.graph_pool_projector = nn.Sequential(
            nn.Linear(2 * gat_hidden_dim, self.graph_pool_dim),
            nn.ReLU(),
            nn.LayerNorm(self.graph_pool_dim),
        )
        self.core_encoder = nn.Sequential(
            nn.Linear(self.core_feature_dim, core_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(core_hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_feature_dim, global_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(global_hidden_dim),
        )

        # [A7_norm, A12_norm, width_norm, A12_fraction, mixed_indicator]
        self.allocation_encoder = nn.Sequential(
            nn.Linear(5, allocation_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(allocation_hidden_dim),
        )

        core_summary_dim = self.num_resource_types * core_hidden_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(
                self.graph_pool_dim + core_summary_dim + global_hidden_dim,
                context_hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(context_hidden_dim),
        )

        pair_feature_dim = (
            gat_hidden_dim + allocation_hidden_dim + context_hidden_dim
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(pair_feature_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1),
        )

        critic_input_dim = (
            self.graph_pool_dim + core_summary_dim + global_hidden_dim
        )
        self.value_head = nn.Sequential(
            nn.Linear(critic_input_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )
        self._initialize_dense_layers()

    def _initialize_dense_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(
                    module.weight,
                    gain=nn.init.calculate_gain("relu"),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_action = self.action_scorer[-1]
        nn.init.orthogonal_(final_action.weight, gain=0.01)
        nn.init.zeros_(final_action.bias)

        final_value = self.value_head[-1]
        nn.init.orthogonal_(final_value.weight, gain=1.0)
        nn.init.zeros_(final_value.bias)

    @staticmethod
    def _ensure_batch_dimension(
        tensor: torch.Tensor,
        expected_ndim_without_batch: int,
    ) -> torch.Tensor:
        if tensor.ndim == expected_ndim_without_batch:
            return tensor.unsqueeze(0)
        return tensor

    def _build_batched_edge_index(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_index = self._ensure_batch_dimension(edge_index, 2).long()
        edge_mask = self._ensure_batch_dimension(edge_mask, 1).bool()
        node_mask = self._ensure_batch_dimension(node_mask, 1).bool()

        batch_size = edge_index.shape[0]
        source = edge_index[:, 0, :]
        target = edge_index[:, 1, :]
        in_range = (
            (source >= 0)
            & (source < self.max_tasks)
            & (target >= 0)
            & (target < self.max_tasks)
        )
        safe_source = source.clamp(0, self.max_tasks - 1)
        safe_target = target.clamp(0, self.max_tasks - 1)
        source_real = torch.gather(node_mask, 1, safe_source)
        target_real = torch.gather(node_mask, 1, safe_target)
        valid = edge_mask & in_range & source_real & target_real

        offsets = (
            torch.arange(
                batch_size,
                device=edge_index.device,
                dtype=torch.long,
            )
            * self.max_tasks
        ).unsqueeze(1)
        source = (source + offsets).reshape(-1)
        target = (target + offsets).reshape(-1)
        valid = valid.reshape(-1)

        if bool(valid.any()):
            return torch.stack([source[valid], target[valid]], dim=0)
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    def _encode_graph(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_features = self._ensure_batch_dimension(
            observations["node_features"], 2
        ).float()
        node_mask = self._ensure_batch_dimension(observations["node_mask"], 1).bool()
        edge_index = self._ensure_batch_dimension(observations["edge_index"], 2)
        edge_mask = self._ensure_batch_dimension(observations["edge_mask"], 1)

        batch_size = node_features.shape[0]
        flat_mask = node_mask.reshape(batch_size * self.max_tasks)

        node_embeddings = self.node_input(node_features)
        node_embeddings = node_embeddings * node_mask.unsqueeze(-1).to(
            node_embeddings.dtype
        )
        node_embeddings = node_embeddings.reshape(
            batch_size * self.max_tasks,
            -1,
        )
        batched_edge_index = self._build_batched_edge_index(
            edge_index,
            edge_mask,
            node_mask,
        )
        for block in self.gat_blocks:
            node_embeddings = block(
                node_embeddings,
                batched_edge_index,
                flat_mask,
            )
        node_embeddings = node_embeddings.reshape(batch_size, self.max_tasks, -1)

        # ------------------------------------------------------------
        # Learnable Mean + Max graph pooling
        # ------------------------------------------------------------
        # node_mask prevents padded/non-existent task slots from affecting
        # either statistic. Shape conventions:
        #   node_embeddings: [B, N, H]
        #   node_mask      : [B, N]
        #   masked_mean    : [B, H]
        #   masked_max     : [B, H]
        #   pooled_stats   : [B, 2H]
        #   graph_embedding: [B, graph_pool_dim]
        mask = node_mask.unsqueeze(-1)
        mask_float = mask.to(node_embeddings.dtype)

        valid_node_count = mask_float.sum(dim=1).clamp_min(1.0)
        masked_sum = (node_embeddings * mask_float).sum(dim=1)
        masked_mean = masked_sum / valid_node_count

        # Invalid/padded nodes must never win the max operation.
        # finfo.min is used instead of -inf to avoid propagating infinities
        # into later dense layers in pathological inputs.
        very_negative = torch.finfo(node_embeddings.dtype).min
        masked_for_max = node_embeddings.masked_fill(~mask, very_negative)
        masked_max = masked_for_max.max(dim=1).values

        # Defensive fallback for a malformed observation containing no real
        # nodes. A valid DAG should always contain at least one node.
        has_real_node = node_mask.any(dim=1, keepdim=True)
        masked_max = torch.where(
            has_real_node,
            masked_max,
            torch.zeros_like(masked_max),
        )

        pooled_stats = torch.cat([masked_mean, masked_max], dim=-1)
        graph_embedding = self.graph_pool_projector(pooled_stats)
        return node_embeddings, graph_embedding, node_mask

    def _allocation_features(
        self,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build one composition feature vector per (task, slot).

        node_features[..., 2] is m_i / max_gang_size by environment contract.
        """
        batch_size = node_features.shape[0]
        width_norm = node_features[:, :, 2].clamp(0.0, 1.0)

        slots = torch.arange(
            self.num_action_slots,
            device=node_features.device,
            dtype=node_features.dtype,
        )
        a12_norm = slots / max(float(self.max_gang_size), 1.0)
        a12_norm = a12_norm.view(1, 1, self.num_action_slots).expand(
            batch_size,
            self.max_tasks,
            self.num_action_slots,
        )

        width_expanded = width_norm.unsqueeze(-1).expand_as(a12_norm)
        a7_norm = (width_expanded - a12_norm).clamp_min(0.0)
        a12_fraction = a12_norm / width_expanded.clamp_min(1e-8)
        mixed = ((a7_norm > 1e-8) & (a12_norm > 1e-8)).to(node_features.dtype)

        return torch.stack(
            [
                a7_norm,
                a12_norm,
                width_expanded,
                a12_fraction,
                mixed,
            ],
            dim=-1,
        )

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embeddings, graph_embedding, _ = self._encode_graph(observations)
        node_features = self._ensure_batch_dimension(
            observations["node_features"], 2
        ).float()
        core_features = self._ensure_batch_dimension(
            observations["core_features"], 2
        ).float()
        global_features = self._ensure_batch_dimension(
            observations["global_features"], 1
        ).float()

        core_embeddings = self.core_encoder(core_features)
        core_summary = core_embeddings.reshape(core_embeddings.shape[0], -1)
        global_embedding = self.global_encoder(global_features)

        context = self.context_encoder(
            torch.cat(
                [graph_embedding, core_summary, global_embedding],
                dim=-1,
            )
        )

        allocation_features = self._allocation_features(node_features)
        allocation_embeddings = self.allocation_encoder(allocation_features)

        batch_size = node_embeddings.shape[0]
        node_for_pairs = node_embeddings.unsqueeze(2).expand(
            batch_size,
            self.max_tasks,
            self.num_action_slots,
            node_embeddings.shape[-1],
        )
        context_for_pairs = context[:, None, None, :].expand(
            batch_size,
            self.max_tasks,
            self.num_action_slots,
            context.shape[-1],
        )
        pair_features = torch.cat(
            [node_for_pairs, allocation_embeddings, context_for_pairs],
            dim=-1,
        )
        action_logits = self.action_scorer(pair_features).squeeze(-1)
        action_logits = action_logits.reshape(
            batch_size,
            self.max_tasks * self.num_action_slots,
        )

        value_input = torch.cat(
            [graph_embedding, core_summary, global_embedding],
            dim=-1,
        )
        values = self.value_head(value_input)
        return action_logits, values


class MaskedGATActorCriticPolicy(ActorCriticPolicy):
    """
    PPO policy with directed GAT, Gang-composition scoring, action masking,
    and alpha exploration mixed INSIDE the categorical distribution.

    P_mixed = (1-alpha) * P_policy + alpha * P_uniform_valid
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args: Any,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 2,
        core_hidden_dim: int = 32,
        global_hidden_dim: int = 32,
        allocation_hidden_dim: int = 32,
        context_hidden_dim: int = 64,
        graph_pool_dim: int = 128,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
        attention_dropout: float = 0.10,
        exploration_alpha: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.gat_hidden_dim = int(gat_hidden_dim)
        self.gat_heads = int(gat_heads)
        self.gat_layers = int(gat_layers)
        self.core_hidden_dim = int(core_hidden_dim)
        self.global_hidden_dim = int(global_hidden_dim)
        self.allocation_hidden_dim = int(allocation_hidden_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.graph_pool_dim = int(graph_pool_dim)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.exploration_alpha = 0.0
        self.set_exploration_alpha(exploration_alpha)

        kwargs["features_extractor_class"] = _UnusedDictExtractor
        kwargs["features_extractor_kwargs"] = {}
        kwargs["net_arch"] = []
        kwargs["ortho_init"] = False

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )

    def set_exploration_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("exploration_alpha must be between 0 and 1.")
        self.exploration_alpha = alpha

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError("MaskedGATActorCriticPolicy requires Dict observations.")
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("MaskedGATActorCriticPolicy requires Discrete actions.")

        self.scheduler_net = GraphAttentionGangSchedulerNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gat_hidden_dim=self.gat_hidden_dim,
            gat_heads=self.gat_heads,
            gat_layers=self.gat_layers,
            core_hidden_dim=self.core_hidden_dim,
            global_hidden_dim=self.global_hidden_dim,
            allocation_hidden_dim=self.allocation_hidden_dim,
            context_hidden_dim=self.context_hidden_dim,
            graph_pool_dim=self.graph_pool_dim,
            actor_hidden_dim=self.actor_hidden_dim,
            critic_hidden_dim=self.critic_hidden_dim,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    @staticmethod
    def _safe_action_mask(
        logits: torch.Tensor,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_mask = observations.get("action_mask")
        if action_mask is None:
            return torch.ones_like(logits, dtype=torch.bool)
        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.bool()
        has_valid = action_mask.any(dim=1, keepdim=True)
        return torch.where(
            has_valid,
            action_mask,
            torch.ones_like(action_mask),
        )

    def _distribution_and_value(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[Distribution, torch.Tensor]:
        action_logits, values = self.scheduler_net(observations)
        safe_mask = self._safe_action_mask(action_logits, observations)
        masked_logits = action_logits.masked_fill(~safe_mask, -1e9)
        distribution = self.action_dist.proba_distribution(action_logits=masked_logits)

        alpha = float(self.exploration_alpha)
        if alpha > 0.0:
            base_probabilities = distribution.distribution.probs
            valid_float = safe_mask.to(base_probabilities.dtype)
            uniform_valid = valid_float / valid_float.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
            mixed_probabilities = (
                (1.0 - alpha) * base_probabilities + alpha * uniform_valid
            )
            mixed_probabilities = mixed_probabilities * valid_float
            mixed_probabilities = mixed_probabilities / mixed_probabilities.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            distribution = self.action_dist.proba_distribution(
                action_logits=torch.log(mixed_probabilities.clamp_min(1e-8))
            )
        return distribution, values

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probability = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_probability

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        distribution, values = self._distribution_and_value(observations)
        actions = actions.long().flatten()
        log_probability = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_probability, entropy

    def get_distribution(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Distribution:
        distribution, _ = self._distribution_and_value(observations)
        return distribution

    def predict_values(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        _, values = self.scheduler_net(observations)
        return values

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "gat_hidden_dim": self.gat_hidden_dim,
                "gat_heads": self.gat_heads,
                "gat_layers": self.gat_layers,
                "core_hidden_dim": self.core_hidden_dim,
                "global_hidden_dim": self.global_hidden_dim,
                "allocation_hidden_dim": self.allocation_hidden_dim,
                "context_hidden_dim": self.context_hidden_dim,
                "graph_pool_dim": self.graph_pool_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
                "exploration_alpha": self.exploration_alpha,
            }
        )
        return data
