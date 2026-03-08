from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import Any, Dict, Sequence, Optional
import flax.linen as nn
import chex


@chex.dataclass
class EmbeddingConfig:
    obj_vocab_size: int
    obj_emb_dim: int

    ingredient_vocab_size: int
    ingredient_emb_dim: int


class ActorCriticBase(ABC, nn.Module):
    action_dim: Sequence[int]
    config: Dict
    # embedding_config: Optional[EmbeddingConfig]

    def __init__(
        self,
        action_dim: Sequence[int],
        config: Dict,
        # embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.action_dim = action_dim
        self.config = config
        # self.embedding_config = embedding_config

    @abstractmethod
    def __call__(self, hidden: Any, x: Any, train: bool) -> (Any, Any, jnp.ndarray):
        """
        Perform a forward pass through the model.

        Args:
            hidden (Any): The hidden state of the RNN.
            x (Any): The input to the model, typically observation and done flag.

        Returns:
            tuple: A tuple containing the new hidden state, the policy distribution, and the value estimate.
        """
        pass

    # def _embed(self, x):
    #     if self.embedding_config is None:
    #         return x

    #     obs, done = x

    #     obj_layer = obs[:, :, :, :, 0]
    #     ingredient_layer = obs[:, :, :, :, 1]

    #     # print("obj_layer: ", obj_layer.shape)
    #     # print("ingredient_layer: ", ingredient_layer.shape)

    #     # print("obj_vocab type: ", self.embedding_config.obj_vocab_size, type(self.embedding_config.obj_vocab_size))

    #     # print("obj_layer", obj_layer)

    #     obj_emb = nn.Embed(
    #         num_embeddings=self.embedding_config.obj_vocab_size,
    #         features=self.embedding_config.obj_emb_dim,
    #         name="obj_embedding",
    #     )(obj_layer)

    #     ingredient_emb = nn.Embed(
    #         num_embeddings=self.embedding_config.ingredient_vocab_size,
    #         features=self.embedding_config.ingredient_emb_dim,
    #         name="ingredient_embedding",
    #     )(ingredient_layer)

    #     obs_emb = jnp.concatenate([obj_emb, ingredient_emb], axis=-1)

    #     return obs_emb, done
