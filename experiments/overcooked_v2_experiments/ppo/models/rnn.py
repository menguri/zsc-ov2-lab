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
from .e3t import PartnerPredictionModule, ScannedSTLPartnerPredictor


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

    @staticmethod
    def initialize_carry(batch_size, hidden_size, action_dim=6):
        rnn_carry = ScannedRNN.initialize_carry(batch_size, hidden_size)
        z_carry = jnp.zeros((batch_size, action_dim))
        return (rnn_carry, z_carry)

    @nn.compact
    def __call__(self, hidden, x, train=False, partner_prediction=None, obs_history=None, act_history=None):
        # Unpack hidden state
        if isinstance(hidden, tuple):
            rnn_state, z_state = hidden
        else:
            rnn_state = hidden
            z_state = jnp.zeros((hidden.shape[0], self.action_dim))

        obs, dones = x

        # E3T / STL 추론 로직
        # partner_prediction이 None이거나, 초기화(init)를 위해 obs_history가 있으면 실행
        if obs_history is not None:
            # Expand dims to add Time dimension for Scan (T_scan=1)
            # obs_history: (Batch, Context, H, W, C) -> (1, Batch, Context, H, W, C)
            # 만약 이미 6차원이라면 (Time, Batch, Context, ...) 그대로 둠
            if obs_history.ndim == 5:
                obs_history_seq = jnp.expand_dims(obs_history, axis=0)
            else:
                obs_history_seq = obs_history
            
            if act_history is None:
                # act_history가 없으면 0으로 초기화 (Batch, Context)
                # obs_history_seq가 (1, B, C, ...) 형태이므로 B, C 추출
                B = obs_history_seq.shape[1]
                C = obs_history_seq.shape[2]
                act_history = jnp.zeros((B, C), dtype=jnp.int32)
            
            # act_history: (Batch, Context) -> (1, Batch, Context)
            if act_history.ndim == 2:
                act_history_seq = jnp.expand_dims(act_history, axis=0)
            else:
                act_history_seq = act_history
            
            # STL Prediction
            anchor_val = self.config.get("anchor", False)
            # obs_history_seq.shape[1] is Batch
            B = obs_history_seq.shape[1]
            # anchor_seq: (1, Batch)
            # Time 차원(1)에 맞춰 확장
            T_scan = obs_history_seq.shape[0]
            anchor_seq = jnp.full((T_scan, B), anchor_val, dtype=bool)
            
            stl_in = (obs_history_seq, act_history_seq, anchor_seq)
            # Use name="shared_predictor" to share parameters with predict_partner
            # Run the predictor to ensure params are initialized or to get prediction
            new_z_state, generated_prediction = ScannedSTLPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_state, stl_in)
            
            # Remove Time dimension: (1, Batch, Dim) -> (Batch, Dim)
            # Scan 결과는 항상 Time 차원을 포함하므로, 단일 스텝인 경우 제거
            if obs_history.ndim == 5:
                new_z_state = new_z_state[0]
                generated_prediction = generated_prediction[0]
            
            if partner_prediction is None:
                z_state = new_z_state
                partner_prediction = generated_prediction

        # print("cnn shapes", rnn_state.shape, obs.shape, dones.shape)

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )

        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        # E3T Conditioning (Layer 4)
        if partner_prediction is not None:
            # Defensive check: squeeze extra dimension if present (e.g. (Batch, 1, Time, Dim))
            if partner_prediction.ndim > embedding.ndim:
                partner_prediction = partner_prediction.squeeze(axis=1)
            embedding = jnp.concatenate([embedding, partner_prediction], axis=-1)

        rnn_in = (embedding, dones)
        rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
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
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return (rnn_state, z_state), pi, jnp.squeeze(critic, axis=-1)

    @nn.compact
    def predict_partner(self, obs_history, act_history, z_state=None, anchor=False):
        """
        E3T / STL Partner Prediction
        Args:
            obs_history: (Batch, 5, H, W, C)
            act_history: (Batch, 5)
            z_state: (Batch, ActionDim) - Optional, for STL
            anchor: bool - Optional, for STL
        Returns:
            partner_prediction: (Batch, ActionDim)
        """
        # If z_state is not provided (e.g. initialization), use zeros
        batch_size = obs_history.shape[0]
        if z_state is None:
            z_state = jnp.zeros((batch_size, self.action_dim))
            
        # Add Time dimension for Scanned module: (1, Batch, ...)
        obs_history_seq = obs_history[jnp.newaxis, ...]
        act_history_seq = act_history[jnp.newaxis, ...]
        anchor_seq = jnp.full((1, batch_size), anchor, dtype=bool)
        
        stl_in = (obs_history_seq, act_history_seq, anchor_seq)
        
        # Use name="shared_predictor" to share parameters with __call__
        # We ignore the new z_state here as this method is for prediction output only
        _, partner_prediction_seq = ScannedSTLPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_state, stl_in)
        
        # Remove Time dimension: (1, Batch, Dim) -> (Batch, Dim)
        result = partner_prediction_seq[0]
        # If result has extra dimension (Batch, 1, Time, Dim), squeeze it
        if result.ndim == 4 and result.shape[1] == 1:
            result = result.squeeze(axis=1)
        return result

    @nn.compact
    def predict_partner_trajectory(self, obs_history, act_history, z_init=None, anchor=False):
        """
        E3T / STL Partner Prediction for Trajectory (Scan)
        Args:
            obs_history: (T, Batch, Context, H, W, C)
            act_history: (T, Batch, Context)
            z_init: (Batch, ActionDim) - Optional
            anchor: bool
        Returns:
            partner_prediction: (T, Batch, ActionDim)
        """
        # obs_history가 (T, B, Context, ...) 형태인지 확인
        # 만약 (T, B, H, W, C)라면 Context 차원이 누락된 것일 수 있음 (주의)
        # 하지만 여기서는 호출자가 올바르게 준다고 가정 (ippo.py에서 처리됨)
        
        T, B = obs_history.shape[:2]
        if z_init is None:
            z_init = jnp.zeros((B, self.action_dim))
            
        anchor_seq = jnp.full((T, B), anchor, dtype=bool)
        stl_in = (obs_history, act_history, anchor_seq)
        
        # Use name="shared_predictor" to share parameters
        _, partner_prediction = ScannedSTLPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_init, stl_in)
        
        return partner_prediction
