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
    """
    Minimal extractor required by ActorCriticPolicy's constructor.

    The real graph processing is performed by GraphAttentionSchedulerNetwork.
    This extractor is intentionally not used by the overridden policy methods.
    """

    def __init__(self, observation_space: gym.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return observations["global_features"].float()[:, :1]


class DirectedGATBlock(nn.Module):
    """
    One bidirectional graph-attention block.

    The original DAG edges carry predecessor -> successor information.
    A second GATConv receives the reversed edges so every task can also
    aggregate information from its successors. The two directions are fused,
    followed by a residual connection and layer normalization.
    """

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        attention_dropout: float,
        feature_dropout: float,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if heads <= 0:
            raise ValueError("heads must be positive.")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

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


class GraphAttentionSchedulerNetwork(nn.Module):
    """
    GAT actor-critic network for DAG scheduling.

    Actor:
        1. Encode every task with bidirectional GAT layers.
        2. Encode every processor core independently.
        3. Create a dedicated score for every (task, core) pair.

    Critic:
        Pool the graph embeddings and combine them with core/global state to
        estimate one scalar state value.
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
        context_hidden_dim: int = 64,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
        attention_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        if not isinstance(observation_space, gym.spaces.Dict):
            raise TypeError("GraphAttentionSchedulerNetwork requires Dict observations.")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("GraphAttentionSchedulerNetwork requires a Discrete action space.")
        if gat_layers <= 0:
            raise ValueError("gat_layers must be at least 1.")
        if gat_hidden_dim % gat_heads != 0:
            raise ValueError("gat_hidden_dim must be divisible by gat_heads.")

        node_shape = observation_space["node_features"].shape
        core_shape = observation_space["core_features"].shape
        global_shape = observation_space["global_features"].shape

        self.max_tasks = int(node_shape[0])
        self.node_feature_dim = int(node_shape[1])
        self.num_cores = int(core_shape[0])
        self.core_feature_dim = int(core_shape[1])
        self.global_feature_dim = int(global_shape[0])

        expected_actions = self.max_tasks * self.num_cores
        if int(action_space.n) != expected_actions:
            raise ValueError(
                "Action space size must equal max_tasks * num_cores: "
                f"expected {expected_actions}, received {action_space.n}."
            )

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

        self.graph_pool_gate = nn.Sequential(
            nn.Linear(gat_hidden_dim, gat_hidden_dim),
            nn.Tanh(),
            nn.Linear(gat_hidden_dim, 1),
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

        self.context_encoder = nn.Sequential(
            nn.Linear(
                gat_hidden_dim + core_hidden_dim + global_hidden_dim,
                context_hidden_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(context_hidden_dim),
        )

        pair_feature_dim = gat_hidden_dim + core_hidden_dim + context_hidden_dim
        self.action_scorer = nn.Sequential(
            nn.Linear(pair_feature_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, 1),
        )

        critic_input_dim = gat_hidden_dim + core_hidden_dim + global_hidden_dim
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
        """Initialize ordinary Linear layers without overriding GAT parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small initial action scores improve early PPO stability.
        final_action_layer = self.action_scorer[-1]
        nn.init.orthogonal_(final_action_layer.weight, gain=0.01)
        nn.init.zeros_(final_action_layer.bias)

        final_value_layer = self.value_head[-1]
        nn.init.orthogonal_(final_value_layer.weight, gain=1.0)
        nn.init.zeros_(final_value_layer.bias)

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
        """
        Convert padded edges shaped (B, 2, E) into one disjoint PyG graph.

        Graph b receives a node-index offset of b * max_tasks. Invalid/padded
        edges and edges pointing to padded nodes are removed.
        """
        edge_index = self._ensure_batch_dimension(edge_index, 2).long()
        edge_mask = self._ensure_batch_dimension(edge_mask, 1).bool()
        node_mask = self._ensure_batch_dimension(node_mask, 1).bool()

        batch_size, _, _ = edge_index.shape
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
        source_is_real = torch.gather(node_mask, 1, safe_source)
        target_is_real = torch.gather(node_mask, 1, safe_target)

        valid = edge_mask & in_range & source_is_real & target_is_real

        offsets = (
            torch.arange(batch_size, device=edge_index.device, dtype=torch.long)
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
            observations["node_features"],
            2,
        ).float()
        node_mask = self._ensure_batch_dimension(
            observations["node_mask"],
            1,
        ).bool()
        edge_index = self._ensure_batch_dimension(
            observations["edge_index"],
            2,
        )
        edge_mask = self._ensure_batch_dimension(
            observations["edge_mask"],
            1,
        )

        batch_size = node_features.shape[0]
        flat_node_mask = node_mask.reshape(batch_size * self.max_tasks)

        node_embeddings = self.node_input(node_features)
        node_embeddings = node_embeddings * node_mask.unsqueeze(-1).to(node_embeddings.dtype)
        node_embeddings = node_embeddings.reshape(batch_size * self.max_tasks, -1)

        pyg_edge_index = self._build_batched_edge_index(
            edge_index=edge_index,
            edge_mask=edge_mask,
            node_mask=node_mask,
        )

        for gat_block in self.gat_blocks:
            node_embeddings = gat_block(
                node_embeddings=node_embeddings,
                edge_index=pyg_edge_index,
                node_mask=flat_node_mask,
            )

        node_embeddings = node_embeddings.reshape(batch_size, self.max_tasks, -1)

        pool_logits = self.graph_pool_gate(node_embeddings).squeeze(-1)
        pool_logits = pool_logits.masked_fill(~node_mask, -1e9)
        pool_weights = torch.softmax(pool_logits, dim=1)
        pool_weights = pool_weights * node_mask.to(pool_weights.dtype)
        pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        graph_embedding = torch.sum(
            node_embeddings * pool_weights.unsqueeze(-1),
            dim=1,
        )

        return node_embeddings, graph_embedding, node_mask

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embeddings, graph_embedding, _ = self._encode_graph(observations)

        core_features = self._ensure_batch_dimension(
            observations["core_features"],
            2,
        ).float()
        global_features = self._ensure_batch_dimension(
            observations["global_features"],
            1,
        ).float()

        core_embeddings = self.core_encoder(core_features)
        global_embedding = self.global_encoder(global_features)
        core_summary = core_embeddings.mean(dim=1)

        context = self.context_encoder(
            torch.cat(
                [graph_embedding, core_summary, global_embedding],
                dim=-1,
            )
        )

        batch_size = node_embeddings.shape[0]
        node_for_pairs = node_embeddings.unsqueeze(2).expand(
            batch_size,
            self.max_tasks,
            self.num_cores,
            node_embeddings.shape[-1],
        )
        core_for_pairs = core_embeddings.unsqueeze(1).expand(
            batch_size,
            self.max_tasks,
            self.num_cores,
            core_embeddings.shape[-1],
        )
        context_for_pairs = context[:, None, None, :].expand(
            batch_size,
            self.max_tasks,
            self.num_cores,
            context.shape[-1],
        )

        pair_features = torch.cat(
            [node_for_pairs, core_for_pairs, context_for_pairs],
            dim=-1,
        )
        action_logits = self.action_scorer(pair_features).squeeze(-1)
        action_logits = action_logits.reshape(batch_size, self.max_tasks * self.num_cores)

        value_input = torch.cat(
            [graph_embedding, core_summary, global_embedding],
            dim=-1,
        )
        values = self.value_head(value_input)

        return action_logits, values


class MaskedGATActorCriticPolicy(ActorCriticPolicy):
    """
    PPO policy with a real graph-attention encoder and action masking.

    The policy preserves the environment's action encoding:
        action = task * num_cores + core
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
        context_hidden_dim: int = 64,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.10,
        attention_dropout: float = 0.10,
        **kwargs: Any,
    ) -> None:
        self.gat_hidden_dim = int(gat_hidden_dim)
        self.gat_heads = int(gat_heads)
        self.gat_layers = int(gat_layers)
        self.core_hidden_dim = int(core_hidden_dim)
        self.global_hidden_dim = int(global_hidden_dim)
        self.context_hidden_dim = int(context_hidden_dim)
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)

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

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise TypeError("MaskedGATActorCriticPolicy requires a Dict observation space.")
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("MaskedGATActorCriticPolicy requires a Discrete action space.")

        self.scheduler_net = GraphAttentionSchedulerNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            gat_hidden_dim=self.gat_hidden_dim,
            gat_heads=self.gat_heads,
            gat_layers=self.gat_layers,
            core_hidden_dim=self.core_hidden_dim,
            global_hidden_dim=self.global_hidden_dim,
            context_hidden_dim=self.context_hidden_dim,
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
    def _apply_action_mask(
        logits: torch.Tensor,
        observations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        action_mask = observations.get("action_mask")
        if action_mask is None:
            return logits

        if action_mask.ndim == 1:
            action_mask = action_mask.unsqueeze(0)
        action_mask = action_mask.bool()

        # Terminal observations can have no valid action. Keep logits finite in
        # that exceptional case; normal non-terminal states remain fully masked.
        has_valid_action = action_mask.any(dim=1, keepdim=True)
        safe_mask = torch.where(
            has_valid_action,
            action_mask,
            torch.ones_like(action_mask),
        )

        return logits.masked_fill(~safe_mask, -1e9)

    def _distribution_and_value(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[Distribution, torch.Tensor]:
        action_logits, values = self.scheduler_net(observations)
        action_logits = self._apply_action_mask(action_logits, observations)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution, values

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, values = self._distribution_and_value(observations)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        distribution, values = self._distribution_and_value(observations)
        actions = actions.long().flatten()
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

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
                "context_hidden_dim": self.context_hidden_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
            }
        )
        return data
