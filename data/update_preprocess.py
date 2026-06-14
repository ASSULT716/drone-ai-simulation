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
BASE_RL_DATA_DIR = 'data\\RL_data'        # 원본 데이터 루트
OUTPUT_DIR       = 'data\\processed_data' # 기본 NPZ 저장 폴더
BACKUP_ROOT_DIR  = 'data\\RL_data_backup'  # 통합 백업 루트 폴더

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

MAX_TIME_GAP = 0.15    
MAX_STEPS    = 10000   

# ==============================================================================
# 2. Packet (.bin) 파일 전담 독립 복사 엔진 ✨
# ==============================================================================
def sync_packet_folders():
    """RL_data/{Difficulty}/Packet/*.bin 파일을 감시하여 백업 폴더로 무조건 복사합니다."""
    # 원본 Packet 폴더 하위의 모든 bin 파일 수집
    bin_files = Path(BASE_RL_DATA_DIR).glob('*/Packet/*.bin')
    
    for bin_path in bin_files:
        try:
            difficulty = bin_path.parent.parent.name  # Easy 또는 Hard 추출
            backup_packet_dir = Path(BACKUP_ROOT_DIR) / difficulty / 'Packet'
            backup_bin_path = backup_packet_dir / bin_path.name

            # 이미 백업 폴더에 복사본이 존재한다면 통과
            if backup_bin_path.exists():
                continue

            # 파일이 완전히 쓰여질 때까지 크기 체크 (I/O 가로채기 방지)
            size_before = bin_path.stat().st_size
            time.sleep(0.1)
            size_after = bin_path.stat().st_size
            
            if size_before != size_after or size_after == 0:
                continue  # 아직 작성 중이면 다음 루프에 처리

            # 백업 폴더 생성 및 복사 실행
            backup_packet_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(bin_path), str(backup_bin_path))
            print(f"  [Packet 복사 완료] {difficulty}/Packet/{bin_path.name} -> 백업 폴더 이동 완료")

        except Exception as e:
            print(f"  [Packet 복사 에러] {bin_path.name}: {e}")

# ==============================================================================
# 3. CSV 전처리 및 복사 엔진 (BIN 복사 로직을 제거하여 가볍게 유지)
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
    try:
        difficulty = csv_path.parent.parent.name  
        filename   = csv_path.stem
        
        backup_csv_dir = Path(BACKUP_ROOT_DIR) / difficulty / 'CSV'
        backup_npz_dir = Path(BACKUP_ROOT_DIR) / difficulty / 'processed_data'
        
        backup_csv_path = backup_csv_dir / csv_path.name
        npz_path        = Path(OUTPUT_DIR) / f'{difficulty}_{filename}.npz'
        backup_npz_path = backup_npz_dir / f'{difficulty}_{filename}.npz'

        # 중복 처리 방지
        if backup_csv_path.exists():
            return True

        df_clean = load_and_sanitize(str(csv_path))

        # 가비지 데이터 필터링 시 CSV만 원본 보존용 복사
        if len(df_clean) <= 5:
            backup_csv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(csv_path), str(backup_csv_path))
            return True

        buffer = build_replay_buffer(df_clean)

        if len(buffer['states']) == 0:
            backup_csv_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(csv_path), str(backup_csv_path))
            return True

        # 정상 저장 및 복사
        np.savez_compressed(str(npz_path), **buffer)
        backup_csv_dir.mkdir(parents=True, exist_ok=True)
        backup_npz_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy(str(csv_path), str(backup_csv_path))
        shutil.copy(str(npz_path), str(backup_npz_path))

        print(f"  [{difficulty.upper()}] {filename}.csv 전처리 및 백업 완료 (.csv / .npz)")
        return True

    except Exception as e:
        print(f"  [CSV 오류] {csv_path.name}: {e}")
        return False

# ==============================================================================
# 4. 실시간 메인 루프
# ==============================================================================
def watch_and_process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BACKUP_ROOT_DIR, exist_ok=True)

    print(f"[실시간 독립 멀티 백업 시스템 시작]")
    print(f"  감시 경로: {BASE_RL_DATA_DIR}")
    print(f"  통합 백업: {BACKUP_ROOT_DIR}\n")

    processed   = set()  
    error_files = set()  

    while True:
        # 🔥 1단계: Packet 폴더에 쌓인 .bin 파일들 독립적으로 싹 긁어서 복사
        sync_packet_folders()

        # 2단계: CSV 폴더 감시 및 전처리 진행
        csv_files = sorted(Path(BASE_RL_DATA_DIR).glob('*/CSV/*.csv'))

        for csv_path in csv_files:
            if csv_path in processed or csv_path in error_files:
                continue

            try:
                size_before = csv_path.stat().st_size
                time.sleep(0.3)  
                size_after = csv_path.stat().st_size

                if size_before != size_after or size_after == 0:
                    continue  

                success = process_one_csv(csv_path)
                if success:
                    processed.add(csv_path)
                else:
                    error_files.add(csv_path)

            except Exception as e:
                error_files.add(csv_path)

        time.sleep(1.0)  

if __name__ == '__main__':
    watch_and_process()