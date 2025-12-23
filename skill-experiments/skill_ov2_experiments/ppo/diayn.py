import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from typing import Sequence, Any

class Discriminator(nn.Module):
    """
    DIAYN Discriminator Network
    상태(s')를 입력받아 스킬(z)의 확률 분포(logits)를 예측합니다.
    """
    num_skills: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        # x: (batch, obs_dim...)
        # 이미지가 입력인 경우 Flatten
        x = x.reshape((x.shape[0], -1))
        
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        logits = nn.Dense(self.num_skills)(x)
        return logits

class DIAYNState(TrainState):
    """
    Discriminator의 파라미터와 옵티마이저 상태를 저장하는 클래스
    """
    pass

def create_diayn_state(rng, obs_shape, num_skills, learning_rate, hidden_dims=(256, 256)):
    """
    DIAYNState 초기화 함수
    """
    discriminator = Discriminator(num_skills=num_skills, hidden_dims=hidden_dims)
    
    # 더미 입력 생성 (배치 차원 포함)
    # obs_shape가 (H, W, C)라면 (1, H, W, C)로 생성
    dummy_obs = jnp.zeros((1, *obs_shape))
    
    params = discriminator.init(rng, dummy_obs)
    
    tx = optax.adam(learning_rate)
    
    return DIAYNState.create(
        apply_fn=discriminator.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def calculate_intrinsic_reward(discriminator_state: DIAYNState, next_obs, skills, num_skills):
    """
    DIAYN Intrinsic Reward 계산
    r = log q_phi(z|s') - log p(z)
    
    Args:
        discriminator_state: 학습된 Discriminator 상태
        next_obs: 다음 상태 관측값 (batch, ...)
        skills: 현재 에이전트에게 할당된 스킬 인덱스 (batch, )
        num_skills: 전체 스킬 개수 (p(z) 계산용)
    
    Returns:
        intrinsic_reward: (batch, )
    """
    # 1. q(z|s') 예측 (Logits)
    logits = discriminator_state.apply_fn(discriminator_state.params, next_obs)
    
    # 2. Log Softmax로 log q(z|s') 계산
    log_probs = jax.nn.log_softmax(logits)
    
    # 3. 실제 할당된 스킬(z)에 대한 log probability 추출
    # skills는 정수 인덱스라고 가정
    batch_indices = jnp.arange(skills.shape[0])
    log_q_z_s = log_probs[batch_indices, skills]
    
    # 4. Prior p(z)는 Uniform 가정: p(z) = 1/num_skills
    # log p(z) = -log(num_skills)
    log_p_z = -jnp.log(num_skills)
    
    # 5. Intrinsic Reward = log q(z|s') - log p(z)
    intrinsic_reward = log_q_z_s - log_p_z
    
    return intrinsic_reward

@jax.jit
def update_discriminator(discriminator_state: DIAYNState, batch_next_obs, batch_skills):
    """
    Discriminator 업데이트 (Cross Entropy Loss)
    Minimize -log q(z|s')
    
    Args:
        discriminator_state: 현재 Discriminator 상태
        batch_next_obs: 배치 관측값
        batch_skills: 배치 스킬 라벨 (정수 인덱스)
        
    Returns:
        new_state: 업데이트된 상태
        loss: 계산된 손실 값
    """
    def loss_fn(params):
        logits = discriminator_state.apply_fn(params, batch_next_obs)
        # optax의 softmax_cross_entropy_with_integer_labels 사용
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_skills)
        return loss.mean()
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(discriminator_state.params)
    new_state = discriminator_state.apply_gradients(grads=grads)
    
    return new_state, loss

def augment_obs_with_skill(obs, skill_idx, num_skills):
    """
    관측값에 스킬 정보를 주입하는 헬퍼 함수 (Broadcasting 방식)
    
    Args:
        obs: (H, W, C) 형태의 관측값
        skill_idx: 스킬 인덱스 (Scalar)
        num_skills: 전체 스킬 개수
        
    Returns:
        augmented_obs: (H, W, C + num_skills)
    """
    H, W, C = obs.shape
    skill_one_hot = jax.nn.one_hot(skill_idx, num_skills) # (num_skills,)
    
    # 공간 차원으로 브로드캐스팅 (H, W, num_skills)
    skill_map = jnp.tile(skill_one_hot, (H, W, 1))
    
    return jnp.concatenate([obs, skill_map], axis=-1)
