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
from .e3t import PartnerPredictionModule, ScannedPartnerPredictor


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
    def encode_obs(self, obs):
        """
        Helper method to get observation embedding without running the full RNN/Partner prediction.
        Useful for PH-1 penalty calculation or simple inference.
        """
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="shared_encoder",
        )
        shared_ln = nn.LayerNorm(name="shared_encoder_ln")

        # Encode current observation
        # If input has Time dimension (T, B, ...), vmap over it.
        if obs.ndim == 5:
            obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        else:
            obs_emb = shared_ln(embed_model(obs))
            
        return obs_emb

    @nn.compact
    def encode_blocked(self, blocked_states):
        """Encode blocked target ($\
        tilde{s}$) into latent space.

        This encoder is intentionally *separate* from `encode_obs`, because the
        execution observation may have different channel semantics (and even
        different channel count) from the global full state used for PH1.

        Args:
            blocked_states: (B, H, W, C_full) or (T, B, H, W, C_full)
        Returns:
            blocked_emb: (B, D) or (T, B, D)
        """
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        blocked_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
            name="blocked_encoder",
        )
        blocked_ln = nn.LayerNorm(name="blocked_encoder_ln")

        if blocked_states.ndim == 5:
            return blocked_ln(jax.vmap(blocked_model)(blocked_states))
        return blocked_ln(blocked_model(blocked_states))

    @nn.compact
    def get_obs_embedding(self, obs):
        # Keep old method alias just in case, but redirect to encode_obs
        return self.encode_obs(obs)

    @nn.compact
    def __call__(
        self,
        hidden,
        x,
        train=False,
        partner_prediction=None,
        obs_history=None,
        act_history=None,
        blocked_states=None,
        agent_idx=None,
    ):
        # NOTE: `agent_idx` is accepted for backward compatibility with older
        # training scripts, but is intentionally ignored (PH1 agent index
        # conditioning was removed).
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
            # STL Removed
            predictor_in = (obs_history_seq, act_history_seq)
            
            # Use name="shared_predictor" to share parameters with predict_partner
            # Run the predictor to ensure params are initialized or to get prediction
            new_z_state, generated_prediction = ScannedPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_state, predictor_in)
            
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
            name="shared_encoder",
        )
        shared_ln = nn.LayerNorm(name="shared_encoder_ln")

        # Encode current observation
        # embedding shape: (T, B, H, W, C) -> (T, B, D)
        obs_emb = shared_ln(jax.vmap(embed_model)(obs))
        embedding = obs_emb

        # E3T Conditioning (Layer 4)
        if partner_prediction is not None:
            # Defensive check: squeeze extra dimension if present (e.g. (Batch, 1, Time, Dim))
            if partner_prediction.ndim > embedding.ndim:
                partner_prediction = partner_prediction.squeeze(axis=1)
            embedding = jnp.concatenate([embedding, partner_prediction], axis=-1)

        # [STA-PH1] blocked_states가 이미지(상태/관측)인 경우 인코딩
        # NOTE: blocked target is expected to be a *global full* state, which may
        # have different channel count from the execution observation.
        blocked_emb = None
        blocked_emb_slots = None
        if blocked_states is not None:
            blocked_states_in = blocked_states.astype(jnp.float32)

            # 이미지 형태 판별: (B,H,W,C) 또는 (T,B,H,W,C)
            # NOTE:
            #   PH1의 blocked target(tilde{s})은 full state일 수 있어
            #   execution obs(agent_view_size 적용)와 spatial shape가 달라도 정상이다.
            #   따라서 obs와의 H/W shape 비교를 하지 않는다.
            #   (좌표 기반 stablock은 ndim<4 이므로 아래 else 경로로 감)
            is_image_like = blocked_states_in.ndim >= 4

            if is_image_like:
                blocked_single = None
                blocked_multi = None

                # Single target path:
                #  - (B,H,W,C) -> (T,B,H,W,C)
                #  - (T,B,H,W,C) -> 그대로
                if blocked_states_in.ndim == obs.ndim - 1:
                    blocked_single = jnp.broadcast_to(
                        blocked_states_in[jnp.newaxis, ...],
                        (obs.shape[0],) + blocked_states_in.shape,
                    )
                elif blocked_states_in.ndim == obs.ndim:
                    if (
                        blocked_states_in.shape[0] == obs.shape[0]
                        and blocked_states_in.shape[1] == obs.shape[1]
                    ):
                        blocked_single = blocked_states_in
                    elif blocked_states_in.shape[0] == obs.shape[1]:
                        # (B,K,H,W,C) with missing time -> multi target
                        blocked_multi = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (obs.shape[0],) + blocked_states_in.shape,
                        )
                elif blocked_states_in.ndim == obs.ndim + 1:
                    # Multi target:
                    #  - (B,K,H,W,C) -> (T,B,K,H,W,C)
                    #  - (T,B,K,H,W,C) -> 그대로
                    if (
                        blocked_states_in.shape[0] == obs.shape[0]
                        and blocked_states_in.shape[1] == obs.shape[1]
                    ):
                        blocked_multi = blocked_states_in
                    elif blocked_states_in.shape[0] == obs.shape[1]:
                        blocked_multi = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (obs.shape[0],) + blocked_states_in.shape,
                        )

                if blocked_multi is not None:
                    # Encode each slot independently, then concatenate slot embeddings.
                    t_dim, b_dim, k_dim = blocked_multi.shape[:3]
                    flat_multi = blocked_multi.reshape(
                        (t_dim, b_dim * k_dim) + blocked_multi.shape[3:]
                    )
                    blocked_emb_flat = self.encode_blocked(flat_multi)
                    blocked_emb_slots = blocked_emb_flat.reshape(
                        (t_dim, b_dim, k_dim, blocked_emb_flat.shape[-1])
                    )
                    blocked_emb = blocked_emb_slots.reshape(
                        (t_dim, b_dim, k_dim * blocked_emb_slots.shape[-1])
                    )
                    embedding = jnp.concatenate([embedding, blocked_emb], axis=-1)
                elif blocked_single is not None:
                    blocked_emb = self.encode_blocked(blocked_single)
                    embedding = jnp.concatenate([embedding, blocked_emb], axis=-1)
            else:
                # 좌표 기반(stablock 기존) 경로는 유지 (Dense input 등)
                if blocked_states_in.ndim == 2:
                    blocked_states_in = blocked_states_in[jnp.newaxis, ...]
                elif blocked_states_in.ndim != embedding.ndim:
                    # 차원 불일치 시 예외 처리 보다는 브로드캐스팅 시도
                    if blocked_states_in.shape[0] == embedding.shape[1]: 
                         # (B, D) -> (1, B, D) -> (T, B, D)
                         blocked_states_in = jnp.broadcast_to(
                            blocked_states_in[jnp.newaxis, ...],
                            (embedding.shape[0],) + blocked_states_in.shape
                        )

                embedding = jnp.concatenate([embedding, blocked_states_in], axis=-1)

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

        # [STA-PH1] Return extras
        extras = {
            "obs_emb": obs_emb,
            "blocked_emb": blocked_emb,
            "blocked_emb_slots": blocked_emb_slots,
        }

        return (rnn_state, z_state), pi, jnp.squeeze(critic, axis=-1), extras

    @nn.compact
    def predict_partner(self, obs_history, act_history, z_state=None):
        """
        E3T Partner Prediction
        Args:
            obs_history: (Batch, 5, H, W, C)
            act_history: (Batch, 5)
            z_state: (Batch, ActionDim) - Optional
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
        
        predictor_in = (obs_history_seq, act_history_seq)
        
        # Use name="shared_predictor" to share parameters with __call__
        # We ignore the new z_state here as this method is for prediction output only
        _, partner_prediction_seq = ScannedPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_state, predictor_in)
        
        # Remove Time dimension: (1, Batch, Dim) -> (Batch, Dim)
        result = partner_prediction_seq[0]
        # If result has extra dimension (Batch, 1, Time, Dim), squeeze it
        if result.ndim == 4 and result.shape[1] == 1:
            result = result.squeeze(axis=1)
        return result

    @nn.compact
    def predict_partner_trajectory(self, obs_history, act_history, z_init=None):
        """
        E3T Partner Prediction for Trajectory (Scan)
        Args:
            obs_history: (T, Batch, Context, H, W, C)
            act_history: (T, Batch, Context)
            z_init: (Batch, ActionDim) - Optional
        Returns:
            partner_prediction: (T, Batch, ActionDim)
        """
        # obs_history가 (T, B, Context, ...) 형태인지 확인
        # 만약 (T, B, H, W, C)라면 Context 차원이 누락된 것일 수 있음 (주의)
        # 하지만 여기서는 호출자가 올바르게 준다고 가정 (ippo.py에서 처리됨)
        
        T, B = obs_history.shape[:2]
        if z_init is None:
            z_init = jnp.zeros((B, self.action_dim))
            
        predictor_in = (obs_history, act_history)
        
        # Use name="shared_predictor" to share parameters
        _, partner_prediction = ScannedPartnerPredictor(action_dim=self.action_dim, name="shared_predictor")(z_init, predictor_in)
        
        return partner_prediction
