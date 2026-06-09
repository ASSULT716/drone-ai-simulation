import pandas as pd
import numpy as np
import os
import glob

# ══════════════════════════════════════════════════════════════════════════════
# 1. 경로 및 설정
# ══════════════════════════════════════════════════════════════════════════════
INPUT_DIR  = 'D:\\drone-ai-simulation\\data\\raw_data'       # 원본 CSV 파일들이 들어있는 폴더
OUTPUT_DIR = 'D:\\drone-ai-simulation\\data\\processed_data' # NPZ 파일들이 저장될 폴더

# 69개 원본 CSV 컬럼 순서
ALL_COLS = [
    'Episode_Count', 'Step_Count', 'Elapsed_Time',
    'Dist_to_Max_X', 'Dist_to_Min_X', 'Dist_to_Max_Y', 'Dist_to_Min_Y', 'Dist_to_Ceiling',
    'Velocity_X', 'Velocity_Y', 'Velocity_Z',
    'Roll', 'Pitch', 'Yaw',
    'Move_Up', 'Move_Right', 'Move_Forward', 'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn',
    'Enemy1_Type', 'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 'Enemy1_Threat_Level',
    'Enemy2_Type', 'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z', 'Enemy2_Threat_Level',
    'Enemy3_Type', 'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z', 'Enemy3_Threat_Level',
    'Lidar_15',  'Lidar_30',  'Lidar_45',  'Lidar_60',  'Lidar_75',  'Lidar_90',
    'Lidar_105', 'Lidar_120', 'Lidar_135', 'Lidar_150', 'Lidar_165', 'Lidar_180',
    'Lidar_195', 'Lidar_210', 'Lidar_225', 'Lidar_240', 'Lidar_255', 'Lidar_270',
    'Lidar_285', 'Lidar_300', 'Lidar_315', 'Lidar_330', 'Lidar_345', 'Lidar_360',
    'Lidar_Down_90',
    'Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch',
    'Cmd_Engine', 'Is_On_Bool', 'Is_Ready_To_Fly_Bool', 'Engine_Power',
    'Step_Reward', 'Done_State',
]

# State(45차원) / Action(6차원) 구성 정의
PHYSICS_FEATURES  = ['Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Roll', 'Pitch', 'Yaw']
ENEMY_FEATURES    = ['Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z',
                     'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z',
                     'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z']
LIDAR_H_FEATURES  = [f'Lidar_{i * 15}' for i in range(1, 25)]
LIDAR_D_FEATURES  = ['Lidar_Down_90']
DEST_FEATURES     = ['Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch']
BOOL_FEATURES     = ['Is_On_Bool', 'Is_Ready_To_Fly_Bool']

STATE_FEATURES    = (PHYSICS_FEATURES + ENEMY_FEATURES +
                     LIDAR_H_FEATURES + LIDAR_D_FEATURES +
                     DEST_FEATURES + BOOL_FEATURES)

ACTION_FEATURES   = ['Move_Up', 'Move_Right', 'Move_Forward',
                     'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn']

# 스케일링을 위한 하드웨어 및 물리적 한계 상수 (분모)
MAX_SPEED      = 5500.0    # cm/s (약 200km/h 기준)
MAX_TIME_GAP   = 0.15      # 초 (프레임 누락 한계)
MAX_STEPS      = 5000      # 스텝 (에피소드 최대 길이)
LIDAR_H_MAX    = 40000.0   # cm
LIDAR_D_MAX    = 5000.0    # cm
MAX_ENEMY_DIST = 400000.0  # cm
MAX_DEST_DIST  = 200000.0  # cm
REWARD_MAX     = 200.0     # 보상

# ══════════════════════════════════════════════════════════════════════════════
# 2. 전처리 파이프라인 함수
# ══════════════════════════════════════════════════════════════════════════════

def load_and_sanitize(csv_path: str) -> pd.DataFrame:
    """1단계: CSV 로드, 엔진 보정, 데이터 타입(불리언 포함) 수치화"""
    df = pd.read_csv(csv_path, header=None, names=ALL_COLS)

    # Cmd_Engine 보정: 처음 '1(On)'이 나오기 전의 '2(Off)'는 초기화 버그로 간주해 '0(대기)'으로 치환
    if (df['Cmd_Engine'] == 1).any():
        first_one_idx = (df['Cmd_Engine'] == 1).idxmax()
        mask = (df.index < first_one_idx) & (df['Cmd_Engine'] == 2)
        if mask.sum() > 0:
            df.loc[mask, 'Cmd_Engine'] = 0

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Float32 수치 변환 (학습 속도 향상)
    num_cols = STATE_FEATURES + ACTION_FEATURES + ['Elapsed_Time', 'Step_Reward']
    for col in num_cols:
        if col in df.columns:
            # boolean 컬럼(True/False)도 이 과정에서 1.0과 0.0으로 자동 변환됩니다.
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """2단계: 물리량을 AI가 이해하기 좋은 형태인 [-1, 1] 또는 [0, 1]로 압축"""
    df = df.copy()

    # 속도(200km/h 기준), 자세각(180도)
    for col in ['Velocity_X', 'Velocity_Y', 'Velocity_Z']:
        df[col] = np.clip(df[col], -MAX_SPEED, MAX_SPEED) / MAX_SPEED
    for col in ['Roll', 'Pitch', 'Yaw']:
        df[col] = df[col] / 180.0

    # 거리 관련 (적 위치, 수평/하향 라이다)
    for col in ENEMY_FEATURES:
        df[col] = np.clip(df[col], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
    for col in LIDAR_H_FEATURES:
        df[col] = np.clip(df[col], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX
    df['Lidar_Down_90'] = np.clip(df['Lidar_Down_90'], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX

    # 목적지 상대 위치 (거리 및 360도 각도 랩핑)
    df['Dest_Rel_Distance'] = np.clip(df['Dest_Rel_Distance'], -MAX_DEST_DIST, MAX_DEST_DIST) / MAX_DEST_DIST
    for col in ['Dest_Rel_Yaw', 'Dest_Rel_Pitch']:
        df[col] = (((df[col] + 180.0) % 360.0) - 180.0) / 180.0

    # 액션(스틱 입력) 및 보상 클리핑
    for col in ACTION_FEATURES:
        df[col] = np.clip(df[col], -1.0, 1.0)
    df['Step_Reward'] = np.clip(df['Step_Reward'], -REWARD_MAX, REWARD_MAX) / REWARD_MAX

    return df

def build_replay_buffer(df: pd.DataFrame) -> dict:
    """3단계: SAC 인공지능 학습을 위한 (상태, 행동, 보상, 다음상태, 종료) 전이 버퍼 조립"""
    s_list, a_list, r_list, s_next_list, term_list, trunc_list = [], [], [], [], [], []

    for _, group in df.groupby('Episode_Count'):
        group = group.sort_values('Step_Count').reset_index(drop=True)

        time_gap = group['Elapsed_Time'].diff().fillna(0.05)
        done_col = group['Done_State']

        # 데이터 필터링: 정상적인 프레임 간격이거나(렉 없음), 게임 종료 스텝인 경우만 통과
        # (과속 이상치 필터링 코드는 데이터 유실 방지를 위해 제거됨)
        valid = (((time_gap > 0) & (time_gap <= MAX_TIME_GAP)) | done_col.isin([1, 2, 3]))

        group = group[valid].reset_index(drop=True)
        if len(group) < 2:
            continue

        states  = group[STATE_FEATURES].values
        actions = group[ACTION_FEATURES].values
        rewards = group['Step_Reward'].values
        dones   = group['Done_State'].values
        steps   = group['Step_Count'].values

        # 마르코프 결정 과정(MDP)에 맞게 (s, a, r, s') 쌍 추출
        for i in range(len(group) - 1):
            is_terminal_next = int(dones[i + 1]) in [1, 2, 3]

            # 중간에 프레임이 튀어 연속된 스텝이 아니면 학습 데이터로 쓰지 않음 (종료 스텝 제외)
            if not is_terminal_next and steps[i + 1] != steps[i] + 1:
                continue

            # 행동(a)을 취해 도달한 결과(s')의 보상을 현재의 보상(r)으로 귀속시킴
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
        'states':      np.array(s_list, dtype=np.float32),
        'actions':     np.array(a_list, dtype=np.float32),
        'rewards':     np.array(r_list, dtype=np.float32),
        'next_states': np.array(s_next_list, dtype=np.float32),
        'terminateds': np.array(term_list, dtype=bool),
        'truncateds':  np.array(trunc_list, dtype=bool),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 3. 메인 실행부 (파일 입출력 자동화)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    
    if not csv_files:
        print(f"[알림] '{INPUT_DIR}' 폴더에 변환할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 드론 비행 로그를 변환합니다.\n")

    for csv_path in sorted(csv_files):
        # 경로에서 파일 이름(확장자 제외)만 추출
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        npz_path = os.path.join(OUTPUT_DIR, f'{filename}.npz')

        print(f"[진행] {filename}.csv 변환 중...")

        # 1. 원시 데이터 세척 및 수치화
        df_clean = load_and_sanitize(csv_path)
        # 2. 정규화 (스케일링)
        df_norm = normalize(df_clean)
        # 3. 리플레이 버퍼 조립
        buffer = build_replay_buffer(df_norm)

        # 4. NPZ 포맷으로 압축 저장 (파일명 동일하게 유지)
        np.savez_compressed(npz_path, **buffer)

    print(f"\n[완료] 변환된 NPZ 파일들이 '{OUTPUT_DIR}'에 저장되었습니다.")
    print(f"       - State 차원: {len(STATE_FEATURES)}")
    print(f"       - Action 차원: {len(ACTION_FEATURES)}")

if __name__ == '__main__':
    main()