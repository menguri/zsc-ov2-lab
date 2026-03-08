# OvercookedV2 Experiment Scripts

이 폴더에는 OvercookedV2 실험을 위한 모든 shell 스크립트들이 포함되어 있습니다.

## 스크립트 목록

### 실험 실행 스크립트
- `run_factory.sh` - 메인 실험 실행 스크립트 (다양한 실험 타입 지원)
- `run_factory_fcp.sh` - FCP (Fixed Checkpoint Population) 실험 전용
- `run_factory_op.sh` - OP (Opponent) 실험 전용
- `run_factory_psp.sh` - PSP (Population-based Self-Play) 실험 전용
- `run_factory_st.sh` - ST (Self-Training) 실험 전용
- `run_user_wandb.sh` - 개별 실험 실행 (Weights & Biases 로깅 포함)

### 시각화 스크립트
- `run_visualize.sh` - 실험 결과 시각화
- `viz_factory.sh` - 배치 시각화 실행
- `viz_helper.sh` - 시각화 헬퍼 함수들

### 유틸리티 스크립트
- `copy_fcp.sh` - FCP population 파일 복사
- `rename_fcp.sh` - FCP 파일 이름 변경
- `train_bc.sh` - Behavioral Cloning 학습
- `c.sh` - 간단한 컴파일/체크 스크립트
- `process_folders.sh` - 폴더 처리 유틸리티

## 사용 방법

### 기본 실행
```bash
# sh_scripts 폴더로 이동
cd sh_scripts

# 실험 실행 예시
./run_factory.sh --exp rnn-op --env grounded_coord_simple

# FCP 실험 실행 예시
./run_factory_fcp.sh --fcp runs/fcp_populations/grounded_coord_simple_avs-2-256-sp
```

### 주요 옵션들
- `--exp`: 실험 타입 (rnn-op, rnn-sa, rnn-fcp, cnn 등)
- `--env`: 환경 타입 (grounded_coord_simple, demo_cook_simple 등)
- `--layout`: 레이아웃 (cramped_room 등)
- `--seeds`: 시드 수 (기본 10)
- `--fcp`: FCP population 경로
- `--gpus`: GPU 할당 (예: --gpus 0,1)

### 팁
- 각 스크립트의 도움말: `./스크립트명.sh --help`
- 환경 변수로 기본값 설정 가능 (예: `export CUDA_VISIBLE_DEVICES=0`)
- W&B API 키는 `../wandb_info/wandb_api_key` 파일에 저장

## 폴더 구조
```
experiments/
├── sh_scripts/          # 모든 shell 스크립트
├── runs/               # 실험 결과
├── fcp_populations/    # FCP population 데이터
├── outputs/            # 추가 출력 파일들
└── wandb_info/         # W&B 설정
```

## 📋 완전한 실험 워크플로우

### 1️⃣ 학습 (Training)

#### 기본 실험 실행
```bash
cd sh_scripts

# Self-Play 학습
./run_factory.sh --exp rnn-sa --env grounded_coord_simple --layout cramped_room --seeds 5

# Opponent-Play 학습
./run_factory.sh --exp rnn-op --env grounded_coord_simple --layout cramped_room --seeds 5

# FCP (Fixed Checkpoint Population) 학습
./run_factory_fcp.sh --fcp ../fcp_populations/grounded_coord_simple_avs-2-256-sp --seeds 1
```

#### 고급 옵션들
```bash
# GPU 지정 및 메모리 설정
./run_factory.sh --exp rnn-op --env grounded_coord_simple --gpus 0 --mem-frac 0.8

# Panic 모드 활성화 (특정 스텝부터 랜덤 액션)
./run_factory.sh --exp rnn-op --env grounded_coord_simple --panic --panic-start 50 --panic-duration 30

# FCP 디바이스 설정 (CPU/GPU)
./run_factory.sh --exp rnn-fcp --env grounded_coord_simple --fcp-device gpu
```

### 2️⃣ 평가 (Evaluation)

#### 개별 실험 평가
```bash
# Self-Play 성능 평가 (비디오 생성)
./run_visualize.sh --gpu 0 --dir ../runs/20251124-041504_avniwfdw_grounded_coord_ring_avs-2-256-sp

# Cross-Play 평가 (다른 모델들과의 대결)
./run_visualize.sh --gpu 0 --dir ../runs/20251124-041504_avniwfdw_grounded_coord_ring_avs-2-256-sp --cross --num_seeds 100

# 메트릭만 계산 (비디오 생성 생략)
./run_visualize.sh --gpu 0 --dir ../runs/20251124-041504_avniwfdw_grounded_coord_ring_avs-2-256-sp --no_viz --cross --num_seeds 500
```

#### 배치 평가
```bash
# 여러 실험 폴더 일괄 평가
./viz_factory.sh --pattern "runs/20251124*" --gpu 0 --cross --num_seeds 100 --no_viz

# 특정 날짜 이후 모든 실험 평가
./viz_helper.sh -p "runs/20251124*"
```

### 3️⃣ 결과 정리 (Result Aggregation)

#### SP/XP 성능 요약
```bash
cd ..  # experiments 폴더로 이동

# 기본 요약 (2025년 11월 23일 이후 모든 실험)
python generate_summary.py

# 특정 날짜 범위 지정
python generate_summary.py --start-date 20251124

# 커스텀 출력 파일 지정
python generate_summary.py --output-file my_summary.csv --start-date 20251120
```

#### 결과 해석
생성된 `summary_sp_xp.csv` 파일에는 다음 정보가 포함됩니다:
- `run_name`: 실험 폴더명
- `sp-mean`: Self-Play 평균 성능
- `sp-std`: Self-Play 표준편차
- `xp-mean`: Cross-Play 평균 성능 (자기 자신 제외)
- `xp-std`: Cross-Play 표준편차
- `gap`: SP - XP 차이 (클수록 일반화 성능 낮음)

### 🎯 전체 워크플로우 예시

```bash
# 1. 학습
cd sh_scripts
./run_factory.sh --exp rnn-op --env grounded_coord_simple --seeds 3

# 2. 평가 (학습 완료 대기 후)
./run_visualize.sh --gpu 0 --dir ../runs/$(ls ../runs/ | tail -1) --cross --num_seeds 100 --no_viz

# 3. 결과 정리
cd ..
python generate_summary.py
```

### 📊 결과 분석 팁

1. **SP 성능**: 모델의 절대 성능을 나타냄
2. **XP 성능**: 다른 모델과의 호환성을 나타냄
3. **Gap**: SP - XP가 크면 과적합 가능성 높음
4. **표준편차**: 값이 작을수록 안정적인 성능

### ⚠️ 주의사항

- 실험 실행 전 GPU 메모리와 W&B API 키 확인
- FCP 실험은 population 파일이 필요함
- 평가 시 seed 수를 충분히 크게 설정 (통계적 신뢰성)
- 긴 실험의 경우 tmux/screen 사용 권장