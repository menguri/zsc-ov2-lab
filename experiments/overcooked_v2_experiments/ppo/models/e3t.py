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
        B, T, H, W, C = obs_history.shape
        
        # Step-wise Encoder 적용 (Time 축에 대해 가중치 공유)
        # (Batch * 5, H, W, C)로 펼쳐서 한 번에 처리
        obs_flat = obs_history.reshape(B * T, H, W, C)
        act_flat = act_history.reshape(B * T)
        
        encoder = StepWiseEncoder(action_dim=self.action_dim)
        encoded_steps = encoder(obs_flat, act_flat) # (B*T, 64)
        
        # 다시 (Batch, 5, 64)로 복원
        encoded_steps = encoded_steps.reshape(B, T, -1)
        
        # 6. Concatenate (Index 6): Flatten and concatenate the 5 outputs
        # (Batch, 5 * 64) = (Batch, 320)
        history_repr = encoded_steps.reshape(B, -1)
        
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

class STLPartnerPredictor(nn.Module):
    """
    STL (Stabilizing Trajectories via Inference-Level Locking)이 적용된 파트너 예측 모듈.
    anchor 플래그에 따라 잠재 벡터(z)를 업데이트하거나 기존 E3T 방식을 따릅니다.
    """
    action_dim: int = 6

    @nn.compact
    def __call__(self, prev_z, obs_history, act_history, anchor):
        """
        Args:
            prev_z: 이전 단계의 잠재 벡터 (Batch, ActionDim) - L2 Normalized
            obs_history: (Batch, 5, H, W, C)
            act_history: (Batch, 5)
            anchor: bool scalar (True: STL 적용, False: Standard E3T)
        
        Returns:
            z_locked: 업데이트된 잠재 벡터 (Batch, ActionDim) - L2 Normalized
            prediction: 파트너 행동 예측 (Batch, ActionDim) - z_locked와 동일
        """
        # 유연한 Shape Unpacking: 뒤에서부터 3개(H, W, C)는 확실하므로 나머지를 Batch/Context로 처리
        *batch_dims, H, W, C = obs_history.shape
        
        # batch_dims가 [Batch, Context] 형태일 것임
        # Flatten Batch and Context for Encoder
        # obs_history: (..., H, W, C) -> (Batch*Context, H, W, C)
        obs_flat = obs_history.reshape(-1, H, W, C)
        act_flat = act_history.reshape(-1)
        
        encoder = StepWiseEncoder(action_dim=self.action_dim)
        encoded_steps = encoder(obs_flat, act_flat)
        
        # Reshape back to (Batch, Context, -1)
        # batch_dims의 마지막 차원이 Context라고 가정
        encoded_steps = encoded_steps.reshape(*batch_dims, -1)
        
        # Flatten Context dimension to get History Representation
        # (Batch, Context, Embed) -> (Batch, Context*Embed)
        history_repr = encoded_steps.reshape(*batch_dims[:-1], -1)
        
        x = history_repr
        for _ in range(3):
            x = nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.leaky_relu(x)
            
        x = nn.Dense(features=64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        
        x = nn.Dense(features=self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        
        # L2 Normalize (z_raw)
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        z_raw = x / (norm + 1e-6)
        
        # 2. STL Logic
        # Cosine Similarity: Dot product of normalized vectors
        # prev_z도 normalized 상태라고 가정
        cos_sim = jnp.sum(z_raw * prev_z, axis=-1, keepdims=True) # (Batch, 1)
        
        # Gating Factor alpha
        alpha = nn.relu(cos_sim)
        
        # Update Rule
        # anchor가 True일 때만 STL 적용
        def update_locked(z_r, z_p, a):
            return a * z_r + (1.0 - a) * z_p
            
        z_locked_stl = update_locked(z_raw, prev_z, alpha)
        
        # anchor에 따른 분기
        # anchor가 (Batch, ) 또는 (Batch, Context) 형태일 수 있음
        # z_raw는 (Batch, Context, Embed_Dim) 형태
        
        # Ensure anchor has same rank as z_raw for broadcasting
        ndim_diff = z_raw.ndim - anchor.ndim
        if ndim_diff > 0:
            for _ in range(ndim_diff):
                anchor = jnp.expand_dims(anchor, axis=-1)
            
        z_locked = jnp.where(anchor, z_locked_stl, z_raw)
        
        # 3. L2 Normalize z_locked
        norm_locked = jnp.linalg.norm(z_locked, axis=-1, keepdims=True)
        z_locked = z_locked / (norm_locked + 1e-6)
        
        return z_locked, z_locked


class ScannedSTLPartnerPredictor(nn.Module):
    """
    STLPartnerPredictor를 시간 축(Time Axis)에 대해 스캔(Scan)하는 래퍼 클래스.
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
            carry: prev_z (Batch, ActionDim)
            x: (obs_history, act_history, anchor)
               obs_history: (Batch, 5, H, W, C)
               act_history: (Batch, 5)
               anchor: (Batch,) or Scalar broadcasted
        """
        prev_z = carry
        obs_history, act_history, anchor = x
        
        model = STLPartnerPredictor(action_dim=self.action_dim)
        new_z, prediction = model(prev_z, obs_history, act_history, anchor)
        
        return new_z, prediction
