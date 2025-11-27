# Overcooked-V2 Metrics & Comparison Guide (ex-overcookedv2)

이 저장소의 실험은 `ex-overcookedv2` 내부에서 수행됩니다. 학습 중/후 의미 있게 비교할 수 있도록 핵심 지표와 로컬 집계 방법을 정리했습니다.

## 핵심 학습 지표(업데이트 단위 평균)
- original_reward: 환경의 진짜 보상(Shaping 제외). 최종 비교 기준 1순위.
- combined_reward: shaping 보상을 합한 학습 보상(초기 학습 모니터링용).
- shaped_reward: shaping 항 자체.
- anneal_factor: shaping 선형 스케줄(훈련 후반에는 0에 근접).
- total_loss, value_loss, loss_actor: PPO 주요 손실(수렴/안정성 판단).
- entropy: 정책 엔트로피(너무 빨리 붕괴하지 않는지 확인).
- ratio: 중요도 샘플링 비율(1 근처 유지가 안정적).

OvercookedV2 전용 로깅 항목이 있을 경우(배달 수, 에피소드 길이 등), 다음도 함께 참고:
- deliveries_per_episode, failed_deliveries, collisions(있다면)
- episode_length: 동일 보상에서 더 짧다면 보다 결정적 행동을 의미할 수 있음.

## 어떤 기준으로 비교할까
- 최종(또는 특정 env_step에서의) original_reward 평균±표준편차 — 1순위
- 샘플 효율: 목표 original_reward 도달까지 필요한 env_step
- 안정성: seed 간 분산, entropy/ratio의 급격한 붕괴·일탈 여부

## 런 태깅/그룹핑 팁
- 실험군: sp/op/sa/fcp (self-play/other-play/self-adapt/foreign-competence population)
- 레이아웃: cramped_room, test_time_wide 등
- 모델: rnn/cnn, hidden size, LR schedule
- teammate 조건: baseline/heldout/population size 등

## 로컬 요약(오프라인) — W&B 없이도 비교
스크립트: `experiments/scripts/aggregate_wandb_summaries.py`
- 검색 대상: `./runs/**/wandb/*/files/wandb-summary.json`
- 각 런 폴더의 `.hydra/config.yaml`이 있으면 `cfg.*` 컬럼으로 함께 병합
- 출력: 기본은 `./runs/summaries.csv`, `--out`으로 다른 경로 지정 가능

예시:

```bash
python experiments/scripts/aggregate_wandb_summaries.py --results-root ./runs --out experiments/outputs/metrics/summaries.csv
```

생성된 CSV를 판다스/스프레드시트로 열어 `original_reward`를 중심으로 layout/실험군/seed 별로 필터·정렬해 비교하세요.

## 온라인 대시보드(W&B) 팁(옵션)
- Line: original_reward vs env_step (seed 그룹/평균±표준편차)
- Line: entropy vs env_step (정책 붕괴 확인)
- Line: ratio vs env_step (학습 안정성)
- Bar: 최종 original_reward per layout (sp/op/sa/fcp 비교)
- Table: best original_reward per run (LR, batch sizes, shaping horizon 등 포함)

참고: shaping을 쓸 때는 최종 비교는 original_reward를 기준으로 하세요. combined_reward는 초반 학습 추적용입니다.
