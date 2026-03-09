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


# (1) latent discretization : partner의 경향성(방향기준)을 conditioned로 전달
class DiClusterEncoder(nn.Module):
    """
    DiClusterEncoder
    - obs_history, action_history를 받고, obs에서 partner 위치 kernel만을 가져오기
    - 위치 kernel을 좌표화 (각도 기반으로 16 방향 cluster로 매핑)
    - Ego와 Partner의 이동 각도를 각각 계산
    - z_ego, z_partner 반환 (One-hot encoding)
    """
    num_clusters: int = 16  # 방향 클러스터 수 (16방향)

    @nn.compact
    def __call__(self, obs_history, act_history, env):
        """
        Args:
            obs_history: (Batch, Time, H, W, C) - 관측 이력
            act_history: (Batch, Time) - 행동 이력 (사용하지 않음, 인터페이스 일관성 유지)
            env: str - 환경 타입 ("ov1" 또는 "ov2")
        Returns:
            z_ego: (Batch, NumClusters) - Ego의 방향 기반 one-hot latent
            z_partner: (Batch, NumClusters) - Partner의 방향 기반 one-hot latent
        """
        # 환경에 따른 채널 인덱스 설정
        # OV1: Ego 위치 채널=0, Partner 위치 채널=1
        # OV2: Ego 위치 채널=0, Partner 위치 채널=10
        if env == "ov1":
            ego_idx = 0
            partner_idx = 1
        else:  # ov2
            ego_idx = 0
            partner_idx = 10
        
        # 1. Shape Unpacking (유연한 배치 처리)
        # 뒤에서부터 3개(H, W, C)를 고정으로 두고, 나머지를 batch_dims로 처리
        *batch_dims, H, W, C = obs_history.shape
        
        # Time 차원이 마지막으로 가정
        if len(batch_dims) < 1:
            raise ValueError("DiClusterEncoder는 최소 Time 차원이 필요합니다.")
            
        time_steps = batch_dims[-1]
        
        # (Batch, Time, H, W, C)로 재구성
        obs_reshaped = obs_history.reshape(-1, time_steps, H, W, C)
        
        # 첫 번째와 마지막 관측 추출 (이동 방향 계산용)
        obs_first = obs_reshaped[:, 0, :, :, :]
        obs_last = obs_reshaped[:, -1, :, :, :]
        
        # 2. Ego와 Partner의 위치 추출
        # 관측 채널에서 최대값(1)의 위치를 찾아 좌표로 변환
        
        def get_pos_xy(obs_frame, channel_idx):
            # obs_frame: (Batch, H, W, C)
            target_map = obs_frame[..., channel_idx]  # (Batch, H, W) - 특정 채널 추출
            
            # 공간 차원을 평탄화하여 최대값 인덱스 찾기 (에이전트 위치)
            flat_map = target_map.reshape(target_map.shape[0], -1)  # (Batch, H*W)
            flat_idx = jnp.argmax(flat_map, axis=-1)  # (Batch,)
            
            # 인덱스를 y, x 좌표로 변환 (행렬 unravel)
            y = flat_idx // W
            x = flat_idx % W
            return y, x  # (Batch,), (Batch,)
            
        # Ego 위치 추출
        ego_y0, ego_x0 = get_pos_xy(obs_first, ego_idx)
        ego_y1, ego_x1 = get_pos_xy(obs_last, ego_idx)
        
        # Partner 위치 추출
        part_y0, part_x0 = get_pos_xy(obs_first, partner_idx)
        part_y1, part_x1 = get_pos_xy(obs_last, partner_idx)
        
        # 3. 이동 각도 계산 및 클러스터 매핑
        # 위치 변화로부터 방향 각도를 계산하고, 16개 클러스터로 이산화
        
        def compute_angle_cls(y0, x0, y1, x1):
            dy = y1 - y0  # y 변화량
            dx = x1 - x0  # x 변화량
            angle = jnp.arctan2(dy, dx)  # 각도 계산 (-π ~ π)
            
            # 각도를 [0, 1) 범위로 정규화
            norm_angle = (angle + jnp.pi) / (2 * jnp.pi)
            
            # 클러스터 인덱스로 이산화 (0 ~ num_clusters-1)
            cls_idx = jnp.floor(norm_angle * self.num_clusters).astype(jnp.int32)
            cls_idx = jnp.clip(cls_idx, 0, self.num_clusters - 1)  # 범위 보장
            
            return cls_idx  # (Batch,)
            
        # Ego와 Partner의 클러스터 인덱스 계산
        z_ego_idx = compute_angle_cls(ego_y0, ego_x0, ego_y1, ego_x1)
        z_part_idx = compute_angle_cls(part_y0, part_x0, part_y1, part_x1)
        
        # 4. One-Hot Encoding
        # 클러스터 인덱스를 one-hot 벡터로 변환
        z_ego = jax.nn.one_hot(z_ego_idx, self.num_clusters)  # (Batch, 16)
        z_part = jax.nn.one_hot(z_part_idx, self.num_clusters)  # (Batch, 16)
        
        # 원래 배치 shape(시간 제외)로 재구성
        batch_shape = tuple(batch_dims[:-1])
        z_ego = z_ego.reshape(*batch_shape, self.num_clusters)
        z_part = z_part.reshape(*batch_shape, self.num_clusters)
        
        return z_ego, z_part