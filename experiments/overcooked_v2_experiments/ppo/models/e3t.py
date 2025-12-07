import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant

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
