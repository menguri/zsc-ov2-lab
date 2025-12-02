import pandas as pd
import os
import argparse
from datetime import datetime
from typing import List

# summary_utils.py에서 방금 생성한 함수를 가져옵니다.
from utils.summary_utils import calculate_metrics_for_run

def find_run_folders(runs_dir: str, start_date_str: str) -> List[str]:
    """지정된 시작 날짜 이후의 모든 실행 폴더를 찾습니다."""
    try:
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
    except ValueError:
        print(f"오류: 날짜 형식이 잘못되었습니다. 'YYYYMMDD' 형식으로 입력해주세요 (예: 20251123).")
        return []

    all_folders = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    
    valid_folders = []
    for folder_name in all_folders:
        try:
            # 폴더 이름의 시작 부분에서 날짜를 파싱합니다.
            folder_date_str = folder_name.split('-')[0]
            folder_date = datetime.strptime(folder_date_str, '%Y%m%d')
            if folder_date >= start_date:
                valid_folders.append(os.path.join(runs_dir, folder_name))
        except (ValueError, IndexError):
            # 날짜 형식에 맞지 않는 폴더는 건너뜁니다.
            continue
            
    print(f"총 {len(valid_folders)}개의 유효한 실행 폴더를 찾았습니다.")
    return sorted(valid_folders)

def main(args):
    """메인 실행 함수"""
    run_folders = find_run_folders(args.runs_dir, args.start_date)
    if not run_folders:
        return

    all_results = []
    for run_path in run_folders:
        metrics = calculate_metrics_for_run(run_path)
        if metrics:
            all_results.append(metrics)

    if not all_results:
        print("처리할 데이터가 없습니다.")
        return

    # 결과를 데이터프레임으로 변환
    new_df = pd.DataFrame(all_results)

    # 출력 파일 처리
    output_path = args.output_file
    if os.path.exists(output_path):
        print(f"기존 파일 '{output_path}'에 결과를 덧붙입니다.")
        try:
            existing_df = pd.read_csv(output_path)
            # 기존 데이터와 새 데이터를 합치고 중복(run_name 기준)을 제거 (새 데이터 우선)
            combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['run_name'], keep='last')
            combined_df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"기존 파일을 읽거나 합치는 중 오류 발생: {e}")
            # 오류 발생 시 새 데이터만으로 덮어쓰기 (안전장치)
            new_df.to_csv(output_path, index=False)
    else:
        print(f"새로운 파일 '{output_path}'을(를) 생성합니다.")
        new_df.to_csv(output_path, index=False)

    print("\n--- 최종 요약 ---")
    print(f"총 {len(new_df)}개의 실행에 대한 결과가 '{output_path}'에 저장되었습니다.")
    # 저장된 파일의 상위 5개 행을 보여줍니다.
    final_df = pd.read_csv(output_path)
    print(final_df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Overcooked 실험 결과(SP/XP)를 요약합니다.")
    
    # 기본 runs 디렉토리를 스크립트 위치 기준으로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_runs_dir = os.path.join(script_dir, 'runs')

    parser.add_argument(
        '--runs-dir', 
        type=str, 
        default=default_runs_dir,
        help=f"실행 결과 폴더들이 있는 상위 디렉토리. 기본값: {default_runs_dir}"
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='20251202',
        help="분석을 시작할 날짜 (YYYYMMDD 형식). 기본값: 20251128"
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        default=os.path.join(default_runs_dir, 'summary_sp_xp.csv'),
        help="요약 결과를 저장할 CSV 파일 경로."
    )

    args = parser.parse_args()
    main(args)
