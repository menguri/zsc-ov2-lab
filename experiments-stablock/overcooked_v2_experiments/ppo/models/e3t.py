import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import functools

class StepWiseEncoder(nn.Module):
    """
    E3T 논문의 Table 4에 정의된 Step-wise Encoder.
    각 시간 단계(t)의 관측(obs)과 행동(act)을 처리합니다.
    """
    action_dim: int = 6
    
    @nn.compact
    def __call__(self, obs, act):
        # obs: (Batch, H, W, C)
        # act: (Batch,) - integer indices

        # 1. Conv2D: 25 filters, 5x5 kernel, LeakyReLU
        # OvercookedV2의 맵 크기가 작을 수 있으므로 padding='SAME'을 사용하여 공간 차원을 유지합니다.
        x = nn.Conv(features=25, kernel_size=(5, 5), padding='SAME', kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(obs)
        x = nn.leaky_relu(x)

        # 2. Conv2D: 25 filters, 3x3 kernel, LeakyReLU
        x = nn.Conv(features=25, kernel_size=(3, 3), padding='SAME', kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.leaky_relu(x)

        # 3. Conv2D: 25 filters, 3x3 kernel, LeakyReLU
        x = nn.Conv(features=25, kernel_size=(3, 3), padding='SAME', kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.leaky_relu(x)

        # 4. Concatenate (Index 4)
        # Flatten the CNN output
        x = x.reshape((x.shape[0], -1))
        
        # Embed the act_history[t]
        # 임베딩 차원은 명시되지 않았으나, 필터 수와 유사하게 25로 설정하거나 64로 설정
        # 여기서는 25로 설정 (필터 수와 맞춤)
        act_embed = nn.Embed(num_embeddings=self.action_dim, features=25, embedding_init=orthogonal(jnp.sqrt(2)))(act.astype(jnp.int32))
        
        # [State_Feature, Action_Embedding] 연결
        x = jnp.concatenate([x, act_embed], axis=-1)

        # 5. MLP (Index 5): 3 layers of (Dense 64 + LeakyReLU)
        for _ in range(3):
            x = nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.leaky_relu(x)
            
        return x

class PartnerPredictionModule(nn.Module):
    """
    E3T 논문의 Table 4에 정의된 Partner Prediction Module.
    파트너의 과거 관측과 행동 이력을 바탕으로 파트너의 다음 행동을 예측(임베딩)합니다.
    """
    action_dim: int = 6

    @nn.compact
    def __call__(self, obs_history, act_history):
        """
        Args:
            obs_history: Shape (Batch, 5, H, W, C)
            act_history: Shape (Batch, 5)
        Returns:
            partner_prediction: Shape (Batch, 6) - L2 Normalized
        """
        # 유연한 Shape Unpacking: 뒤에서부터 3개(H, W, C)는 확실하므로 나머지를 Batch/Context로 처리
        *batch_dims, H, W, C = obs_history.shape
        
        # Flatten for step-wise encoder
        obs_flat = obs_history.reshape(-1, H, W, C)
        act_flat = act_history.reshape(-1)
        
        encoder = StepWiseEncoder(action_dim=self.action_dim)
        encoded_steps = encoder(obs_flat, act_flat) # (Total_Batch, 64)
        
        # Reshape back: (..., Context, Features)
        encoded_steps = encoded_steps.reshape(*batch_dims, -1)
        
        # Flatten Context dimension -> (Batch, Context * Features)
        # 마지막 배치 차원(Time/Context)을 Flatten하여 하나의 벡터로 만듦
        # 가정: batch_dims의 마지막 차원이 Time(5) 차원임
        history_repr = encoded_steps.reshape(*batch_dims[:-1], -1)
        
        x = history_repr

        # 7. MLP (Index 7): 3 layers of (Dense 64 + LeakyReLU)
        for _ in range(3):
            x = nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.leaky_relu(x)

        # 8. FC + Tanh (Index 8): Dense 64 + Tanh activation
        x = nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)

        # 9. Output (Index 9): Dense 6 (Action Dim) + L2 Normalize
        x = nn.Dense(features=self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        
        # L2 Normalize
        # x / (||x|| + 1e-6)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        x = x / (norm + 1e-6)
        
        return x

class ScannedPartnerPredictor(nn.Module):
    """
    PartnerPredictionModule을 시간 축(Time Axis)에 대해 스캔(Scan)하는 래퍼 클래스.
    STL(Stabilizing Trajectories) 로직은 제거됨.
    """
    action_dim: int = 6
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """
        Args:
            carry: (Batch, ActionDim) ignored logic-wise but preserved for scan API
            x: tuple of (obs_history, act_history)
               obs_history: (Batch, 5, H, W, C)
               act_history: (Batch, 5)
        Returns:
            new_z: current prediction (Batch, ActionDim)
            prediction: current prediction (Batch, ActionDim)
        """
        # STL 제거: carry(prev_z)는 사용하지 않지만, 다음 스텝의 carry로 전달 (Dummy State)
        
        # x unpacking
        obs_history, act_history = x

        model = PartnerPredictionModule(action_dim=self.action_dim)
        z = model(obs_history, act_history)
        
        return z, z


class PartnerStatePredictionModule(nn.Module):
    """
    State-prediction branch:
      - Encode history into latent z
      - Decode z into (current partner action logits, next partner obs latent z)
    """
    action_dim: int = 6
    z_dim: int = 64

    @nn.compact
    def __call__(self, obs_history, act_history):
        # obs_history: (..., Context, H, W, C)
        # act_history: (..., Context)
        *batch_dims, H, W, C = obs_history.shape

        obs_flat = obs_history.reshape(-1, H, W, C)
        act_flat = act_history.reshape(-1)

        encoder = StepWiseEncoder(action_dim=self.action_dim)
        encoded_steps = encoder(obs_flat, act_flat)  # (Total_Batch, 64)
        encoded_steps = encoded_steps.reshape(*batch_dims, -1)
        history_repr = encoded_steps.reshape(*batch_dims[:-1], -1)

        x = history_repr
        for _ in range(3):
            x = nn.Dense(
                features=64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.leaky_relu(x)

        # Context latent z used as policy input in state-prediction mode.
        z = nn.Dense(
            features=self.z_dim,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        z = nn.tanh(z)

        d = z
        for _ in range(2):
            d = nn.Dense(
                features=64,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(d)
            d = nn.leaky_relu(d)

        action_logits = nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(d)
        next_z = nn.Dense(
            features=self.z_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(d)
        next_z = nn.tanh(next_z)

        return z, action_logits, next_z


class ScannedPartnerStatePredictor(nn.Module):
    """
    Scan wrapper for state-prediction branch.
    Carry holds latent z.
    """
    action_dim: int = 6
    z_dim: int = 64

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        obs_history, act_history = x
        model = PartnerStatePredictionModule(
            action_dim=self.action_dim,
            z_dim=self.z_dim,
        )
        z, action_logits, next_z = model(obs_history, act_history)
        return z, (z, action_logits, next_z)
