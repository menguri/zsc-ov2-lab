<h1 align="center">JaxMARL</h1>

<p align="center">
       <a href="https://pypi.python.org/pypi/jaxmarl">
        <img src="https://img.shields.io/pypi/pyversions/jaxmarl.svg" /></a>
       <a href="https://badge.fury.io/py/jaxmarl">
        <img src="https://badge.fury.io/py/jaxmarl.svg" /></a>
       <a href= "https://github.com/FLAIROx/JaxMARL/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
       <a href= "https://colab.research.google.com/github/FLAIROx/JaxMARL/blob/main/jaxmarl/tutorials/JaxMARL_Walkthrough.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
       <a href= "https://arxiv.org/abs/2311.10090">
        <img src="https://img.shields.io/badge/arXiv-2311.10090-b31b1b.svg" /></a>
       <a href= "https://jaxmarl.foersterlab.com/">
        <img src="https://img.shields.io/badge/docs-green" /></a>
       
</p>

[**Installation**](#install) | [**Quick Start**](#start) | [**Environments**](#environments) | [**Algorithms**](#algorithms) | [**Citation**](#cite)
---

<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/mabrax.png?raw=true" alt="mabrax" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/hanabi.png?raw=true" alt="hanabi" width="20%">
        </div>
        <div class="row" align="centre">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/coin_game.png?raw=true" alt="coin_game" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/jaxnav-ma.gif?raw=true" alt="jaxnav" width="20%">
            <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/smax.gif?raw=true" alt="SMAX" width="20%">
        </div>
    </div>
</div>

## Multi-Agent Reinforcement Learning in JAX

🎉 **Update: JaxMARL was accepted at NeurIPS 2024 on Datasets and Benchmarks Track. See you in Vacouver!**

JaxMARL combines ease-of-use with GPU-enabled efficiency, and supports a wide range of commonly used MARL environments as well as popular baseline algorithms. Our aim is for one library that enables thorough evaluation of MARL methods across a wide range of tasks and against relevant baselines. We also introduce SMAX, a vectorised, simplified version of the popular StarCraft Multi-Agent Challenge, which removes the need to run the StarCraft II game engine. 

For more details, take a look at our [blog post](https://blog.foersterlab.com/jaxmarl/) or our [Colab notebook](https://colab.research.google.com/github/FLAIROx/JaxMARL/blob/main/jaxmarl/tutorials/JaxMARL_Walkthrough.ipynb), which walks through the basic usage.

<h2 name="environments" id="environments">Environments 🌍 </h2>

| Environment | Reference | README | Summary |
| --- | --- | --- | --- |
| 🔴 MPE | [Paper](https://arxiv.org/abs/1706.02275) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe) | Communication orientated tasks in a multi-agent particle world
| 🍲 Overcooked | [Paper](https://arxiv.org/abs/1910.05789) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked) | Fully-cooperative human-AI coordination tasks based on the homonyms video game | 
| 🦾 Multi-Agent Brax | [Paper](https://arxiv.org/abs/2003.06709) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mabrax) | Continuous multi-agent robotic control based on Brax, analogous to Multi-Agent MuJoCo |
| 🎆 Hanabi | [Paper](https://arxiv.org/abs/1902.00506) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/hanabi) | Fully-cooperative partially-observable multiplayer card game |
| 👾 SMAX | Novel | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax) | Simplified cooperative StarCraft micro-management environment |
| 🧮 STORM: Spatial-Temporal Representations of Matrix Games | [Paper](https://openreview.net/forum?id=54F8woU8vhq) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/storm) | Matrix games represented as grid world scenarios
| 🧭 JaxNav | [Paper](https://www.arxiv.org/abs/2408.15099) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/jaxnav) | 2D geometric navigation for differential drive robots
| 🪙 Coin Game | [Paper](https://arxiv.org/abs/1802.09640) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/coin_game) | Two-player grid world environment which emulates social dilemmas
| 💡 Switch Riddle | [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/switch_riddle) | Simple cooperative communication game included for debugging

 
<h2 name="algorithms" id="algorithms">Baseline Algorithms 🦉 </h2>

We follow CleanRL's philosophy of providing single file implementations which can be found within the `baselines` directory. We use Hydra to manage our config files, with specifics explained in each algorithm's README. Most files include `wandb` logging code, this is disabled by default but can be enabled within the file's config.

| Algorithm | Reference | README | 
| --- | --- | --- | 
| IPPO | [Paper](https://arxiv.org/pdf/2011.09533.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO) | 
| MAPPO | [Paper](https://arxiv.org/abs/2103.01955) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO) | 
| IQL | [Paper](https://arxiv.org/abs/1312.5602v1) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) | 
| VDN | [Paper](https://arxiv.org/abs/1706.05296)  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| QMIX | [Paper](https://arxiv.org/abs/1803.11485) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| TransfQMIX | [Paper](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1679.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| SHAQ | [Paper](https://arxiv.org/abs/2105.15013) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| PQN-VDN | [Paper](https://arxiv.org/abs/2407.04811) | [Source](https://github.com/mttga/purejaxql) |

<h2 name="install" id="install">Installation 🧗 </h2>

**Environments** - Before installing, ensure you have the correct [JAX installation](https://github.com/google/jax#installation) for your hardware accelerator. We have tested up to JAX version 0.4.25. The JaxMARL environments can be installed directly from PyPi:

```
pip install jaxmarl 
```

**Algorithms** - If you would like to also run the algorithms, install the source code as follows:

1. Clone the repository:
    ```
    git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
    ```
2. Install requirements:
    ``` 
    pip install -e .[algs]
    export PYTHONPATH=./JaxMARL:$PYTHONPATH
    ```
3. For the fastest start, we reccoment using our Dockerfile, the usage of which is outlined below.

**Development** - If you would like to run our test suite, install the additonal dependencies with:
 `pip install -e .[dev]`, after cloning the repository.

<h2 name="start" id="start">Quick Start 🚀 </h2>

We take inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax) interfaces. You can try out training an agent in our [Colab notebook](https://colab.research.google.com/github/FLAIROx/JaxMARL/blob/main/jaxmarl/tutorials/JaxMARL_Walkthrough.ipynb). Further introduction scripts can be found [here](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/tutorials).

### Basic JaxMARL API  Usage 🖥️

Actions, observations, rewards and done values are passed as dictionaries keyed by agent name, allowing for differing action and observation spaces. The done dictionary contains an additional `"__all__"` key, specifying whether the episode has ended. We follow a parallel structure, with each agent passing an action at each timestep. For asynchronous games, such as Hanabi, a dummy action is passed for agents not acting at a given timestep.

```python 
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Initialise environment.
env = make('MPE_simple_world_comm_v3')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
```

### Dockerfile 🐋
To help get experiments up and running we include a [Dockerfile](https://github.com/FLAIROx/JaxMARL/blob/main/Dockerfile) and its corresponding [Makefile](https://github.com/FLAIROx/JaxMARL/blob/main/Makefile). With Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed, the container can be built with:
```
make build
```
The built container can then be run:
```
make run
```

## Contributing 🔨
Please contribute! Please take a look at our [contributing guide](https://github.com/FLAIROx/JaxMARL/blob/main/CONTRIBUTING.md) for how to add an environment/algorithm or submit a bug report. Our roadmap also lives there.

## OvercookedV2 PANIC Robustness 확장 🍲🔥

> 이 섹션은 본 포크(`experiments/overcooked_v2_experiments/ppo`)에서 추가된 OvercookedV2용 강건성 평가 기능(PANIC)에 대한 설명입니다. 원본 JaxMARL 라이브러리의 기본 배포에는 포함되지 않습니다.

### 목적
실제 협력 상황에서 파트너(팀메이트)가 순간적으로 불안정하거나 잘못된 행동을 할 때 Ego 정책이 얼마나 강건한지(성능 저하 양상, 회복력, 실패 패턴)를 정량화하기 위한 실험적 기능입니다.

### 핵심 아이디어
각 벡터화된 환경의 에피소드마다 한 에이전트를 무작위(2인 환경은 Bernoulli(0.5) → {0,1})로 선택하여, 설정된 에피소드 로컬 스텝 구간(`[panic.start_step, panic.start_step + panic.duration)`) 동안 그 에이전트의 액션을 균일 난수로 교란(override)합니다. 교란된 액션에 대해 PPO의 `log_prob`를 재계산하여 학습 안정성을 유지합니다.

### Hydra 설정 예시
```bash
python run.py \
    +panic.enabled=true \
    +panic.start_step=50 \
    +panic.duration=30 \
    env=overcooked_v2_rnn_sp
```

설정 키:
- `panic.enabled` (bool): 기능 on/off.
- `panic.start_step` (int): 에피소드 로컬 스텝 기준 시작 지점.
- `panic.duration` (int): 활성 지속 길이(스텝 수). 0 또는 음수이면 자동 비활성(no-op).

### 동작 상세
1. 에피소드 시작 시 환경별로 교란 대상 에이전트 인덱스 배열(`panic_partner_indices`)을 샘플링. 비활성 시 -1 저장.
2. 매 스텝에서 `episode_step`가 지정된 구간에 속하고 대상 인덱스가 유효하면 해당 에이전트 액션을 `Uniform(0, num_actions)` 난수로 덮어씀.
3. 교란 후 정책 분포로부터 `log_prob` 재계산 → PPO ratio 일관성 유지.
4. 종료한 에피소드에 대해 PANIC 관련 per-episode 누적치를 총계로 반영 후 리셋.

### 수집/로깅 메트릭 (wandb)
기능이 비활성(`enabled=false` 또는 `duration<=0`)이면 아래 메트릭은 전혀 생성되지 않습니다.

| Key | 정의 |
| --- | --- |
| `panic/episodes_finished` | PANIC 추적 동안 종료된 에피소드 수 합계 |
| `panic/total_actions` | PANIC 창에서 교란된(override 발생) 스텝 총합 |
| `panic/total_reward` | PANIC 활성 구간에서 팀(첫 번째 에이전트) 보상 누적 |
| `panic/total_deliveries` | PANIC 활성 구간에서 올바른 배달(+DELIVERY_REWARD) 횟수 |
| `panic/total_wrong_deliveries` | PANIC 활성 구간에서 잘못된 배달(-DELIVERY_REWARD) 횟수 |
| `panic/actions_per_episode` | `total_actions / max(episodes_finished,1)` |
| `panic/reward_per_episode` | `total_reward / max(episodes_finished,1)` |
| `panic/deliveries_per_episode` | `total_deliveries / max(episodes_finished,1)` |
| `panic/wrong_deliveries_per_episode` | `total_wrong_deliveries / max(episodes_finished,1)` |

### 구현 파일
- `ippo.py`: PANIC 창 활성 판정, 액션 override 적용, per-step/episode 상태 유지.
- `panic_utils.py`: 대상 선택, 액션 교란, 누적/집계 함수 (한국어 라인별 주석 포함).

### 성능/오버헤드
비활성 시 조건문 조기 탈출로 추가 연산/메모리 오버헤드는 최소화됩니다(주로 몇 개의 zero 배열 유지). JIT 컴파일 후 교란 분기 비용은 매우 낮습니다.

### 에지 케이스 & 향후 확장
- `start_step`가 실제 에피소드 길이보다 크면 교란은 발생하지 않음(자동 no-op).
- `duration=0` → 전체 메트릭 비생성.
- N>2 다중 에이전트 환경 확장: 현재 2인 환경은 Bernoulli(0.5), 일반화 시 Uniform over agents 사용 가능 (코드에 fallback 포함).
- 합법/legal 액션 집합 필터링: 향후 환경별 제한 액션 존재 시 샘플링을 전체 공간 대신 합법 집합으로 좁힐 수 있음.

### 연구 활용 시 권장 분석
1. `panic/actions_per_episode` vs 기본 성능 저하율.
2. `panic/wrong_deliveries_per_episode` 급증 구간 탐지 → 파트너 노이즈 민감도.
3. `panic/reward_per_episode` 회복 곡선 추적 → 정책 회복력(resilience) 정량화.

### 간단한 의사코드
```python
if panic_enabled and 0 < duration and start_step <= ep_step < start_step + duration:
        action[target_agent, env_idx] = random.randint(0, num_actions)
        log_prob = pi.log_prob(flatten(action))
```

### 인용/명시
논문 또는 보고서에서 사용 시 “OvercookedV2 PANIC Robustness Extension (uniform partner action perturbation)” 형태로 명시하는 것을 권장합니다.


<h2 name="cite" id="cite">Citing JaxMARL 📜 </h2>
If you use JaxMARL in your work, please cite us as follows:

```
@article{flair2023jaxmarl,
      title={JaxMARL: Multi-Agent RL Environments in JAX},
      author={Alexander Rutherford and Benjamin Ellis and Matteo Gallici and Jonathan Cook and Andrei Lupu and Gardar Ingvarsson and Timon Willi and Akbir Khan and Christian Schroeder de Witt and Alexandra Souly and Saptarashmi Bandyopadhyay and Mikayel Samvelyan and Minqi Jiang and Robert Tjarko Lange and Shimon Whiteson and Bruno Lacerda and Nick Hawes and Tim Rocktaschel and Chris Lu and Jakob Nicolaus Foerster},
      journal={arXiv preprint arXiv:2311.10090},
      year={2023}
    }
```

## See Also 🙌
There are a number of other libraries which inspired this work, we encourage you to take a look!

JAX-native algorithms:
- [Mava](https://github.com/instadeepai/Mava): JAX implementations of IPPO and MAPPO, two popular MARL algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
- [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.
- [JaxIRL](https://github.com/FLAIROx/jaxirl?tab=readme-ov-file): JAX implementation of algorithms for inverse reinforcement learning.

JAX-native environments:
- [Gymnax](https://github.com/RobertTLange/gymnax): Implementations of classic RL tasks including classic control, bsuite and MinAtar.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse set of environments ranging from simple games to NP-hard combinatorial problems.
- [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- [Brax](https://github.com/google/brax): A fully differentiable physics engine written in JAX, features continuous control tasks.
- [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid): Meta-RL gridworld environments inspired by XLand and MiniGrid.
- [Craftax](https://github.com/MichaelTMatthews/Craftax): (Crafter + NetHack) in JAX.