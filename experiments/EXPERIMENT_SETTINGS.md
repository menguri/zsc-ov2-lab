# OvercookedV2 Experiment Settings Summary

## 공통 설정 (Base Config)

```yaml
SEED: 42
NUM_SEEDS: 10                    # Self-Play, Other-Play, State-Action에서 사용
NUM_CHECKPOINTS: 3               # 체크포인트 저장 개수 (초반, 중반, 최종)
VISUALIZE: False                 # 학습 후 자동 시각화 여부
```

---

## 1. CNN (Self-Play) - Original Overcooked

### 모델 구조
```yaml
TYPE: CNN
FC_DIM_SIZE: 64
CNN_FEATURES: 32
ACTIVATION: relu
```

### 학습 하이퍼파라미터
```yaml
# Training Steps
TOTAL_TIMESTEPS: 1e7             # 10M timesteps
REW_SHAPING_HORIZON: 5e6         # 5M (reward shaping이 적용되는 구간)

# PPO Parameters
LR: 4e-4                         # 0.0004
ANNEAL_LR: True                  # Learning rate annealing
LR_WARMUP: 0.05                  # 처음 5%는 warmup

GAMMA: 0.99                      # Discount factor
GAE_LAMBDA: 0.95                 # GAE lambda
VF_COEF: 0.5                     # Value function loss coefficient
ENT_COEF: 0.04                   # Entropy coefficient
MAX_GRAD_NORM: 0.5               # Gradient clipping
CLIP_EPS: 0.2                    # PPO clipping epsilon

# Batch Configuration
NUM_ENVS: 64                     # Parallel environments
NUM_STEPS: 256                   # Steps per environment per update
UPDATE_EPOCHS: 4                 # PPO update epochs
NUM_MINIBATCHES: 16              # Minibatches per update

# Effective batch size: 64 * 256 = 16,384
# Minibatch size: 16,384 / 16 = 1,024
# Total updates: 10M / (64 * 256) = 610
```

---

## 2. RNN (Self-Play)

### 모델 구조
```yaml
TYPE: RNN
FC_DIM_SIZE: 128
GRU_HIDDEN_DIM: 128
CNN_FEATURES: 32
ACTIVATION: relu
```

### 학습 하이퍼파라미터
```yaml
# Training Steps
TOTAL_TIMESTEPS: 3e7             # 30M timesteps
REW_SHAPING_HORIZON: 1.5e7       # 15M

# PPO Parameters
LR: 2.5e-4                       # 0.00025
ANNEAL_LR: True
LR_WARMUP: 0.05

GAMMA: 0.99
GAE_LAMBDA: 0.95
VF_COEF: 0.5
ENT_COEF: 0.01                   # CNN보다 낮음
MAX_GRAD_NORM: 0.25              # CNN보다 낮음
CLIP_EPS: 0.2

# Batch Configuration
NUM_ENVS: 256                    # CNN의 4배
NUM_STEPS: 256
UPDATE_EPOCHS: 4
NUM_MINIBATCHES: 64              # CNN의 4배

# Effective batch size: 256 * 256 = 65,536
# Minibatch size: 65,536 / 64 = 1,024
# Total updates: 30M / (256 * 256) = 458
```

---

## 3. RNN-OP (Other-Play)

### 모델 구조
```yaml
TYPE: RNN
FC_DIM_SIZE: 128
GRU_HIDDEN_DIM: 128
CNN_FEATURES: 32
ACTIVATION: relu
```

### 학습 하이퍼파라미터
```yaml
# Training Steps
TOTAL_TIMESTEPS: 5e7             # 50M timesteps (Self-Play보다 많음)
REW_SHAPING_HORIZON: 2.5e7       # 25M

# PPO Parameters
LR: 2.5e-4                       # 0.00025 (Self-Play와 동일)
ANNEAL_LR: True
LR_WARMUP: 0.05

GAMMA: 0.99
GAE_LAMBDA: 0.95
VF_COEF: 0.5
ENT_COEF: 0.02                   # Self-Play(0.01)보다 높음
MAX_GRAD_NORM: 0.25
CLIP_EPS: 0.2

# Batch Configuration
NUM_ENVS: 64                     # Self-Play보다 작음
NUM_STEPS: 256
UPDATE_EPOCHS: 4
NUM_MINIBATCHES: 64

# Effective batch size: 64 * 256 = 16,384
# Minibatch size: 16,384 / 64 = 256
# Total updates: 50M / (64 * 256) = 3,052
```

### 환경 설정
```yaml
ENV_KWARGS:
  op_ingredient_permutations: [0, 1]  # 재료 순서 permutation
```

---

## 4. RNN-SA (State-Action)

### 모델 구조
```yaml
TYPE: RNN
FC_DIM_SIZE: 128
GRU_HIDDEN_DIM: 128
CNN_FEATURES: 32
ACTIVATION: relu
```

### 학습 하이퍼파라미터
```yaml
# Iterative Training
NUM_ITERATIONS: 10               # 10번의 iteration (Self-Play와 교차 학습)
# 각 iteration당 TOTAL_TIMESTEPS / 10 = 3M timesteps

# Training Steps (per iteration)
TOTAL_TIMESTEPS: 3e7             # 30M timesteps (총합)
REW_SHAPING_HORIZON: 1.5e7       # 15M

# PPO Parameters (Self-Play와 동일)
LR: 2.5e-4
ANNEAL_LR: True
LR_WARMUP: 0.05

GAMMA: 0.99
GAE_LAMBDA: 0.95
VF_COEF: 0.5
ENT_COEF: 0.01
MAX_GRAD_NORM: 0.25
CLIP_EPS: 0.2

# Batch Configuration (Self-Play와 동일)
NUM_ENVS: 256
NUM_STEPS: 256
UPDATE_EPOCHS: 4
NUM_MINIBATCHES: 64
```

---

## 5. RNN-FCP (Fictitious Co-Play)

### 모델 구조
```yaml
TYPE: RNN
FC_DIM_SIZE: 128
GRU_HIDDEN_DIM: 128
CNN_FEATURES: 32
ACTIVATION: relu
```

### 학습 하이퍼파라미터
```yaml
# Seeds
NUM_SEEDS: 1                     # Ego agent 1개만 학습

# Training Steps
TOTAL_TIMESTEPS: 3e7             # 30M timesteps
REW_SHAPING_HORIZON: 1.5e7       # 15M

# PPO Parameters
LR: 7e-4                         # 0.0007 (Self-Play보다 높음)
ANNEAL_LR: True
LR_WARMUP: 0.05

GAMMA: 0.99
GAE_LAMBDA: 0.9                  # Self-Play(0.95)보다 낮음
VF_COEF: 0.5
ENT_COEF: 0.04                   # Self-Play(0.01)보다 높음
MAX_GRAD_NORM: 0.25
CLIP_EPS: 0.2

# Batch Configuration
NUM_ENVS: 32                     # 메모리 제약으로 감소 (원래 256)
NUM_STEPS: 256
UPDATE_EPOCHS: 4
NUM_MINIBATCHES: 64

# Effective batch size: 32 * 256 = 8,192
# Minibatch size: 8,192 / 64 = 128
# Total updates: 30M / (32 * 256) = 3,662
```

### Population 설정
```yaml
FCP: fcp_populations/<layout_name>  # Pre-trained population path
# Population은 Self-Play로 학습된 10개 run의 checkpoint들
# 각 run당 3개 checkpoint (초반, 중반, 최종) → 총 30개 policy
```

---

## 실험별 비교표

| Experiment | Total Steps | LR | NUM_ENVS | Batch Size | ENT_COEF | GAE_λ | Notes |
|------------|-------------|-----|----------|------------|----------|-------|-------|
| **CNN-SP** | 10M | 4e-4 | 64 | 16,384 | 0.04 | 0.95 | Baseline |
| **RNN-SP** | 30M | 2.5e-4 | 256 | 65,536 | 0.01 | 0.95 | 3x longer training |
| **RNN-OP** | 50M | 2.5e-4 | 64 | 16,384 | 0.02 | 0.95 | Ingredient permutation |
| **RNN-SA** | 30M (10 iter) | 2.5e-4 | 256 | 65,536 | 0.01 | 0.95 | Iterative training |
| **RNN-FCP** | 30M | 7e-4 | 32* | 8,192 | 0.04 | 0.9 | With population |

*FCP의 NUM_ENVS는 GPU 메모리 제약으로 32로 설정 (원래 설정은 256)

---

## 실행 명령어 예시

```bash
# Self-Play (RNN)
./run_user_wandb.sh --gpus 4 --env grounded_coord_simple --exp rnn-sp \
  --env-device cpu --nenvs 128 --nsteps 128

# Other-Play (RNN-OP)
./run_user_wandb.sh --gpus 4 --env grounded_coord_simple --exp rnn-op \
  --env-device cpu --nenvs 128 --nsteps 128

# State-Action (RNN-SA)
./run_user_wandb.sh --gpus 4 --env grounded_coord_simple --exp rnn-sa \
  --iterations 10 --env-device cpu --nenvs 128 --nsteps 128

# FCP (RNN-FCP) - NUM_ENVS 감소 필요
./run_user_wandb.sh --gpus 1 --env grounded_coord_simple --exp rnn-fcp \
  --fcp fcp_populations/grounded_coord_simple_avs-2_sp \
  --env-device cpu --nenvs 32 --nsteps 128
```

---

## 주요 차이점 요약

### 1. **학습 길이**
- CNN-SP: 10M (가장 짧음)
- RNN-SP/SA: 30M
- RNN-OP: 50M (가장 김)

### 2. **Learning Rate**
- CNN: 4e-4 (가장 높음)
- RNN-SP/OP/SA: 2.5e-4
- RNN-FCP: 7e-4 (가장 높음, population 학습 위해)

### 3. **Entropy Coefficient**
- RNN-SP: 0.01 (가장 낮음, exploitation 중심)
- RNN-OP: 0.02
- CNN/RNN-FCP: 0.04 (가장 높음, exploration 중심)

### 4. **Batch Size**
- RNN-SP/SA: 65,536 (가장 큼)
- CNN-SP: 16,384
- RNN-OP: 16,384
- RNN-FCP: 8,192 (가장 작음, 메모리 제약)

### 5. **GAE Lambda**
- 대부분: 0.95
- RNN-FCP: 0.9 (더 짧은 horizon)

---

## Notes

1. **Reward Shaping Horizon**: 모든 실험에서 TOTAL_TIMESTEPS의 50% 지점까지 적용
2. **Warmup**: 모든 실험에서 학습의 처음 5% 동안 learning rate warmup 적용
3. **Gradient Clipping**: RNN 계열은 0.25, CNN은 0.5
4. **FCP 메모리 이슈**: 초기화 시 GPU에서 Conv 연산 → cuDNN workspace 필요 → NUM_ENVS 감소 필요
