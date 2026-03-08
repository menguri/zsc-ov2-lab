import pandas as pd
import os
from typing import Dict, Optional
import re

def _is_not_self_play(label: str) -> bool:
    """
    'policy_labels' (예: 'cross-0_1')를 파싱하여 self-play가 아닌지(예: 0 != 1) 확인합니다.
    'cross-0_0'과 같은 self-play는 False를 반환합니다.
    """
    match = re.search(r'cross-(\d+)_(\d+)', str(label))
    if match:
        return match.group(1) != match.group(2)
    return False # 매칭되지 않는 형식은 cross-play로 간주하지 않음

def calculate_metrics_for_run(run_path: str) -> Optional[Dict[str, float]]:
    """
    단일 실행 폴더에 대한 SP 및 XP 성능 지표를 계산합니다.

    Args:
        run_path (str): 분석할 개별 실행 폴더의 경로.

    Returns:
        Optional[Dict[str, float]]: 계산된 지표(sp-mean, sp-std, xp-mean, xp-std, gap)가 포함된 딕셔너리.
                                     필요한 파일이 없으면 None을 반환합니다.
    """
    run_name = os.path.basename(run_path)
    sp_csv_path = os.path.join(run_path, 'reward_summary_sp.csv')
    cross_csv_path = os.path.join(run_path, 'reward_summary_cross.csv')

    # 필수 CSV 파일 존재 여부 확인
    if not os.path.exists(sp_csv_path) or not os.path.exists(cross_csv_path):
        print(f"[{run_name}] 필수 'reward_summary_sp.csv' 또는 'reward_summary_cross.csv' 파일이 없어 건너뜁니다.")
        return None

    try:
        # 1. Self-Play(SP) 성능 계산
        sp_df = pd.read_csv(sp_csv_path)
        if 'total_reward' not in sp_df.columns:
            print(f"[{run_name}] SP 파일에 'total_reward' 열이 없어 건너뜁니다.")
            return None
            
        sp_mean = sp_df['total_reward'].mean()
        sp_std = sp_df['total_reward'].std()

        # 2. Cross-Play(XP) 성능 계산 (자기 자신과의 대결 제외)
        cross_df = pd.read_csv(cross_csv_path)
        if 'total_reward' not in cross_df.columns or 'policy_labels' not in cross_df.columns:
            print(f"[{run_name}] Cross-Play 파일에 'total_reward' 또는 'policy_labels' 열이 없어 건너뜁니다.")
            return None

        # 'policy_labels'를 기준으로 self-play(예: cross-0_0)를 제외
        xp_df = cross_df[cross_df['policy_labels'].apply(_is_not_self_play)]
        
        if xp_df.empty:
            print(f"[{run_name}] Cross-Play 데이터에서 자기 자신을 제외한 파트너가 없어 XP를 계산할 수 없습니다.")
            xp_mean = 0.0
            xp_std = 0.0
        else:
            xp_mean = xp_df['total_reward'].mean()
            xp_std = xp_df['total_reward'].std()

        # 3. Gap 계산 및 모든 값을 정수로 변환
        sp_mean_int = int(round(sp_mean))
        sp_std_int = int(round(sp_std)) if pd.notna(sp_std) else 0
        xp_mean_int = int(round(xp_mean))
        xp_std_int = int(round(xp_std)) if pd.notna(xp_std) else 0
        gap = sp_mean_int - xp_mean_int

        return {
            'run_name': run_name,
            'sp-mean': sp_mean_int,
            'sp-std': sp_std_int,
            'xp-mean': xp_mean_int,
            'xp-std': xp_std_int,
            'gap': gap
        }

    except Exception as e:
        print(f"[{run_name}] 처리 중 오류 발생: {e}")
        return None
