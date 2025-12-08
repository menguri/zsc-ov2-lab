import functools
from typing import Dict, Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import remat
import distrax
from flax.linen.initializers import constant, orthogonal
from .abstract import ActorCriticBase
from .common import CNN, MLP
from .e3t import PartnerPredictionModule


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry            # rnn_state.shape == (batch_size, hidden_size)
        ins, resets = x              # ins는 scan 한 step에 대한 입력

        # --- 여기 수정 ---
        # ins.shape를 쓰지 말고, 현재 hidden state에서 batch / hidden을 읽어온다.
        batch_size, hidden_size = rnn_state.shape

        new_carry = self.initialize_carry(batch_size, hidden_size)

        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            new_carry,
            rnn_state,
        )

        # GRUCell도 hidden_size를 기준으로 정의
        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(ActorCriticBase):

    @nn.compact
    def __call__(self, hidden, x, train=False, partner_prediction=None, obs_history=None, act_history=None):
        obs, dones = x

        # E3T Initialization Helper: Ensure PartnerPredictionModule parameters are initialized
        if obs_history is not None and act_history is not None:
            self.predict_partner(obs_history, act_history)

        # E3T 추론 로직: partner_prediction이 제공되지 않았지만 obs_history가 있는 경우 계산 수행
        if partner_prediction is None and obs_history is not None:
            # act_history가 누락된 경우 더미(0) 사용
            if act_history is None:
                # obs_history shape: (Time, Batch, k, H, W, C)
                # act_history shape should be: (Time, Batch, k)
                s = obs_history.shape
                act_history = jnp.zeros((s[0], s[1], s[2]), dtype=jnp.int32)
            
            # predict_partner는 (Batch, k, ...)를 기대하므로 Time 축에 대해 vmap 적용
            partner_prediction = jax.vmap(self.predict_partner)(obs_history, act_history)

        print("cnn shapes", hidden.shape, obs.shape, dones.shape)

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        # embedding_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )

        # embedding = jax.vmap(
        #     jax.vmap(embedding_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        embed_model = CNN(
            # output_size=self.config["FC_DIM_SIZE"] * 2,
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )

        # embedding = embed_model(embedding)
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        # E3T Conditioning (Layer 4)
        # partner_prediction이 제공되면 현재 상태 특징(embedding)과 연결합니다.
        if partner_prediction is not None:
            # partner_prediction: (Time, Batch, 6)
            # embedding: (Time, Batch, Hidden)
            embedding = jnp.concatenate([embedding, partner_prediction], axis=-1)

        # embedding_1_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )
        # embedding = jax.vmap(
        #     jax.vmap(embedding_1_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        # embedding = nn.Dense(
        #     self.config["GRU_HIDDEN_DIM"],
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(embedding)

        rnn_in = (embedding, dones)
        # print("rnn_in shapes", hidden.shape, rnn_in[0].shape, rnn_in[1].shape)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # embedding = nn.Dense(
        #     self.config["FC_DIM_SIZE"],
        #     kernel_init=orthogonal(jnp.sqrt(2)),
        #     bias_init=constant(0.0),
        # )(embedding)

        # embedding_2_mlp = MLP(
        #     hidden_size=self.config["FC_DIM_SIZE"],
        #     output_size=self.config["FC_DIM_SIZE"],
        #     activation=activation,
        # )
        # embedding = jax.vmap(
        #     jax.vmap(embedding_2_mlp, in_axes=-2, out_axes=-2), in_axes=-2, out_axes=-2
        # )(embedding)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        # actor_mean = nn.BatchNorm(use_running_average=not self.train)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        # critic = nn.BatchNorm(use_running_average=not self.train)(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # print(
        #     "output shapes",
        #     hidden.shape,
        #     pi.logits.shape,
        #     jnp.squeeze(critic, axis=-1).shape,
        # )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @nn.compact
    def predict_partner(self, obs_history, act_history):
        """
        E3T Partner Prediction
        Args:
            obs_history: (Batch, T, H, W, C)
            act_history: (Batch, T)
        Returns:
            partner_prediction: (Batch, ActionDim)
        """
        return PartnerPredictionModule(action_dim=self.action_dim)(obs_history, act_history)
