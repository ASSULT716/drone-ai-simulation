import pandas as pd
import numpy as np
import os
import glob
import time
import shutil  
from pathlib import Path

# ==============================================================================
# 1. 경로 및 설정
# ==============================================================================
BASE_RL_DATA_DIR = 'data\\RL_data'        # RL_data 루트 (*/csv/*.csv 하위를 감시)
OUTPUT_DIR       = 'data\\processed_data' # 기본 NPZ 저장 폴더
BACKUP_ROOT_DIR  = 'data\\RL_data_backup'  # 통합 백업 루트 폴더

# ==============================================================================
# 2. CSV 72개 컬럼 정의 (UE5 생성 순서와 정확히 일치)
# ==============================================================================
ALL_COLS = [
    'Episode_Count', 'Current_step', 'Elapsed_Time',
    'Dist_to_Max_X', 'Dist_to_Min_X', 'Dist_to_Max_Y', 'Dist_to_Min_Y', 'Dist_to_Ceiling',
    'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Roll', 'Pitch', 'Yaw',
    'Move_Up', 'Move_Right', 'Move_Forward', 'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn',
    'Enemy1_Type', 'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 'Enemy1_Threat_Level',
    'Enemy2_Type', 'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z', 'Enemy2_Threat_Level',
    'Enemy3_Type', 'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z', 'Enemy3_Threat_Level',
    'Lidar_15',  'Lidar_30',  'Lidar_45',  'Lidar_60',  'Lidar_75',  'Lidar_90',
    'Lidar_105', 'Lidar_120', 'Lidar_135', 'Lidar_150', 'Lidar_165', 'Lidar_180',
    'Lidar_195', 'Lidar_210', 'Lidar_225', 'Lidar_240', 'Lidar_255', 'Lidar_270',
    'Lidar_285', 'Lidar_300', 'Lidar_315', 'Lidar_330', 'Lidar_345', 'Lidar_360',
    'Lidar_Down_90', 'Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch',
    'Cmd_Engine', 'Is_On_Bool', 'Is_Ready_To_Fly_Bool', 'Engine_Power',
    'Step_Reward', 'Done_State', 'Pressing_I', 'Is_AI_Controlled', 'Is_Pressing_Backspace',
]

# ==============================================================================
# 3. State(62차원) / Action(6차원) 정의 (bin 파일 순서와 동일)
# ==============================================================================
STATE_FEATURES = [
    'Current_step', 'Done_State', 'Step_Reward',
    'Dist_to_Max_X', 'Dist_to_Min_X', 'Dist_to_Max_Y', 'Dist_to_Min_Y', 'Dist_to_Ceiling',
    'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Roll', 'Pitch', 'Yaw',
    'Lidar_Down_90', 'Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch',
    'Is_On_Bool', 'Is_Ready_To_Fly_Bool', 'Pressing_I', 'Is_AI_Controlled', 'Is_Pressing_Backspace',
    'Enemy1_Type', 'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 'Enemy1_Threat_Level',
    'Enemy2_Type', 'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z', 'Enemy2_Threat_Level',
    'Enemy3_Type', 'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z', 'Enemy3_Threat_Level',
] + [f'Lidar_{i * 15}' for i in range(1, 25)]

ACTION_FEATURES = [
    'Move_Up', 'Move_Right', 'Move_Forward',
    'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn',
]

MAX_TIME_GAP = 0.15    # 초
MAX_STEPS    = 10000   # 스텝

# ==============================================================================
# 5. 전처리 함수
# ==============================================================================
def load_and_sanitize(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=ALL_COLS)

    if (df['Cmd_Engine'] == 1).any():
        first_one_idx = (df['Cmd_Engine'] == 1).idxmax()
        mask = (df.index < first_one_idx) & (df['Cmd_Engine'] == 2)
        if mask.sum() > 0:
            df.loc[mask, 'Cmd_Engine'] = 0

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    target_cols = list(set(STATE_FEATURES + ACTION_FEATURES + ['Episode_Count', 'Elapsed_Time']))
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    return df


def build_replay_buffer(df: pd.DataFrame) -> dict:
    s_list, a_list, r_list, s_next_list, term_list, trunc_list = [], [], [], [], [], []

    for _, group in df.groupby('Episode_Count'):
        group = group.sort_values('Current_step').reset_index(drop=True)

        time_gap = group['Elapsed_Time'].diff().fillna(0.05)
        done_col = group['Done_State']

        valid = (((time_gap > 0) & (time_gap <= MAX_TIME_GAP)) | done_col.isin([1, 2, 3]))
        group = group[valid].reset_index(drop=True)

        if len(group) < 2:
            continue

        states  = group[STATE_FEATURES].values
        actions = group[ACTION_FEATURES].values
        rewards = group['Step_Reward'].values
        dones   = group['Done_State'].values
        steps   = group['Current_step'].values

        for i in range(len(group) - 1):
            is_terminal_next = int(dones[i + 1]) in [1, 2, 3]

            if not is_terminal_next and steps[i + 1] != steps[i] + 1:
                continue

            r = float(rewards[i + 1])

            if is_terminal_next:
                term, trunc = True, False
            elif steps[i + 1] >= MAX_STEPS:
                term, trunc = False, True
            else:
                term, trunc = False, False

            s_list.append(states[i])
            a_list.append(actions[i])
            r_list.append(r)
            s_next_list.append(states[i + 1])
            term_list.append(term)
            trunc_list.append(trunc)

    return {
        'states':      np.array(s_list,      dtype=np.float32),
        'actions':     np.array(a_list,      dtype=np.float32),
        'rewards':     np.array(r_list,      dtype=np.float32),
        'next_states': np.array(s_next_list, dtype=np.float32),
        'terminateds': np.array(term_list,   dtype=bool),
        'truncateds':  np.array(trunc_list,  dtype=bool),
    }


def process_one_csv(csv_path: Path) -> bool:
    """CSV 파일 하나를 전처리하고 데이터셋 및 원본 일체를 백업 폴더로 복사합니다."""
    try:
        difficulty = csv_path.parts[-3]  # RL_data/{difficulty}/csv/xxx.csv
        filename   = csv_path.stem
        
        # 1. 대상 목적지 폴더 및 파일 경로 정의
        backup_csv_dir    = Path(BACKUP_ROOT_DIR) / difficulty / 'csv'
        backup_npz_dir    = Path(BACKUP_ROOT_DIR) / difficulty / 'processed_data'
        
        backup_csv_path = backup_csv_dir / csv_path.name
        npz_path        = Path(OUTPUT_DIR) / f'{difficulty}_{filename}.npz'
        backup_npz_path = backup_npz_dir / f'{difficulty}_{filename}.npz'

        # [중복 방지 안전장치] 이미 복사본 CSV가 존재한다면 과거에 완료된 파일이므로 생략
        if backup_csv_path.exists():
            return True

        # 🔥 [개선 1] 바이너리(.bin) 파일 폴더명 자동 대응 후보군 정의
        # 실제 폴더가 'bin'이든, 'packet'이든, 'csv'와 같은 폴더에 있든 다 찾아냅니다.
        possible_bin_paths = [
            csv_path.parent.parent / 'Packet' / f'{filename}.bin',
            csv_path.parent.parent / 'bin' / f'{filename}.bin',
            csv_path.parent / f'{filename}.bin'
        ]

        # 🔥 [개선 2] 타이밍 이슈 방지: UE5가 bin 파일을 생성/작성 중일 수 있으므로 최대 2.5초간 대기하며 탐색
        source_bin_path = None
        for _ in range(5):  # 0.5초 간격으로 최대 5번 루프
            for p in possible_bin_paths:
                if p.exists() and p.stat().st_size > 0:
                    source_bin_path = p
                    break
            if source_bin_path:
                break
            time.sleep(0.5)

        df_clean = load_and_sanitize(str(csv_path))

        # 2. 물리엔진 안정화용 초기화 파일 필터링 (5줄 이하)
        if len(df_clean) <= 5:
            print(f"  [건너뜀/백업] {filename}.csv — 초기 데이터 부족 (원본 보존용 복사만 수행)")
            backup_csv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(csv_path), str(backup_csv_path))
            if source_bin_path:
                backup_packet_dir = Path(BACKUP_ROOT_DIR) / difficulty / source_bin_path.parent.name
                backup_packet_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(source_bin_path), str(backup_packet_dir / source_bin_path.name))
            return True

        buffer = build_replay_buffer(df_clean)

        # 3. 유효 샘플이 없는 에피소드 필터링
        if len(buffer['states']) == 0:
            print(f"  [건너뜀/백업] {filename}.csv — 유효 샘플 없음 (원본 보존용 복사만 수행)")
            backup_csv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(csv_path), str(backup_csv_path))
            if source_bin_path:
                backup_packet_dir = Path(BACKUP_ROOT_DIR) / difficulty / source_bin_path.parent.name
                backup_packet_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(source_bin_path), str(backup_packet_dir / source_bin_path.name))
            return True

        # 4. 정상 에피소드: NPZ 압축 파일 생성
        np.savez_compressed(str(npz_path), **buffer)

        # 5. 전방위 백업 복사 전개
        backup_csv_dir.mkdir(parents=True, exist_ok=True)
        backup_npz_dir.mkdir(parents=True, exist_ok=True)
        
        # (A) CSV 복사
        shutil.copy(str(csv_path), str(backup_csv_path))
        
        # (B) 🔥 BIN 복사 (실제 원본 폴더 이름 구조를 그대로 본떠서 백업 폴더 자동 빌드)
        if source_bin_path:
            backup_packet_dir = Path(BACKUP_ROOT_DIR) / difficulty / source_bin_path.parent.name
            backup_packet_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(source_bin_path), str(backup_packet_dir / source_bin_path.name))
        else:
            print(f"  [경고] {filename}.csv와 매칭되는 .bin 파일을 찾지 못해 bin 복사는 건너뜁니다.")

        # (C) NPZ 복사
        shutil.copy(str(npz_path), str(backup_npz_path))

        print(f"  [{difficulty.upper()}] {filename}.csv 전처리 및 통합 백업 세트 구성 완료")
        return True

    except Exception as e:
        print(f"  [오류] {csv_path.name}: {e}")
        return False


# ==============================================================================
# 6. 실시간 감시 모드
# ==============================================================================
def watch_and_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_ROOT_DIR, exist_ok=True)

    print(f"[실시간 감시 및 스마트 자동 추적 복사 모드 시작]")
    print(f"  감시 경로(보존): {BASE_RL_DATA_DIR}\\*\\csv\\*.csv")
    print(f"  기존 저장 경로: {OUTPUT_DIR}")
    print(f"  통합 백업 경로: {BACKUP_ROOT_DIR}")
    print(f"  State {len(STATE_FEATURES)}차원 / Action {len(ACTION_FEATURES)}차원\n")

    processed   = set()  
    error_files = set()  

    while True:
        csv_files = sorted(Path(BASE_RL_DATA_DIR).glob('*/CSV/*.csv'))

        for csv_path in csv_files:
            if csv_path in processed or csv_path in error_files:
                continue

            try:
                size_before = csv_path.stat().st_size
                time.sleep(0.5)  
                size_after = csv_path.stat().st_size

                if size_before != size_after or size_after == 0:
                    continue  

                success = process_one_csv(csv_path)
                if success:
                    processed.add(csv_path)
                else:
                    error_files.add(csv_path)

            except Exception as e:
                print(f"  [오류] {csv_path.name}: {e}")
                error_files.add(csv_path)

        time.sleep(1.0)  


if __name__ == '__main__':
    watch_and_process()