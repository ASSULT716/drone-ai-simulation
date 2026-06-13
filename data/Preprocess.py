import pandas as pd
import numpy as np
import os
import glob

# ══════════════════════════════════════════════════════════════════════════════
# 1. 경로 및 설정
# ══════════════════════════════════════════════════════════════════════════════
INPUT_DIR  = 'raw_data'
OUTPUT_DIR = 'processed_data'

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

# ══════════════════════════════════════════════════════════════════════════════
# 2. State(56차원) / Action(6차원) 구성 정의
# ══════════════════════════════════════════════════════════════════════════════
PHYSICS_FEATURES    = ['Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Roll', 'Pitch', 'Yaw']

ENEMY_FEATURES      = ['Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z',
                       'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z',
                       'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z']

ENEMY_META_FEATURES = ['Enemy1_Type', 'Enemy1_Threat_Level',
                       'Enemy2_Type', 'Enemy2_Threat_Level',
                       'Enemy3_Type', 'Enemy3_Threat_Level']

LIDAR_H_FEATURES    = [f'Lidar_{i * 15}' for i in range(1, 25)]
LIDAR_D_FEATURES    = ['Lidar_Down_90']

DEST_FEATURES       = ['Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch']

BOOL_FEATURES       = ['Is_On_Bool', 'Is_Ready_To_Fly_Bool']

BOUNDARY_FEATURES   = ['Dist_to_Max_X', 'Dist_to_Min_X',
                       'Dist_to_Max_Y', 'Dist_to_Min_Y',
                       'Dist_to_Ceiling']

STATE_FEATURES      = (PHYSICS_FEATURES    +   # 6
                       ENEMY_FEATURES      +   # 9
                       ENEMY_META_FEATURES +   # 6
                       LIDAR_H_FEATURES    +   # 24
                       LIDAR_D_FEATURES    +   # 1
                       DEST_FEATURES       +   # 3
                       BOOL_FEATURES       +   # 2
                       BOUNDARY_FEATURES)      # 5  → 합계 56차원

ACTION_FEATURES     = ['Move_Up', 'Move_Right', 'Move_Forward',
                       'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn']  # 6차원

# ══════════════════════════════════════════════════════════════════════════════
# 3. 스케일링 상수
# ══════════════════════════════════════════════════════════════════════════════
MAX_SPEED         = 5500.0    # cm/s  | Velocity 정규화 (약 200 km/h)
MAX_TIME_GAP      = 0.15      # 초    | 프레임 드랍 판단 기준
MAX_STEPS         = 7000      # 스텝  | 에피소드 최대 길이
LIDAR_H_MAX       = 40000.0   # cm    | 수평 라이다 최대 감지 거리
LIDAR_D_MAX       = 5000.0    # cm    | 하향 라이다 최대 감지 거리
MAX_ENEMY_DIST    = 400000.0  # cm    | 적 상대 좌표 정규화
MAX_DEST_DIST     = 200000.0  # cm    | 목적지 거리/벡터 정규화
MAX_THREAT_LEVEL  = 100.0     # 점    | 위협 수준 정규화 (실측 최대 100)
MAX_ENEMY_TYPE    = 3.0       # 종류  | 적 타입 정규화 (1:사람 / 2:자폭 드론 / 3:재머)
MAX_BOUNDARY_XY   = 125000.0  # cm    | X·Y 맵 경계 거리 정규화 (실측 최대 ~125,000 cm)
MAX_CEILING       = 43000.0   # cm    | 천장 거리 정규화 (맵 높이 430m 반영)
REWARD_MAX        = 200.0     # 점    | 보상 정규화

# ══════════════════════════════════════════════════════════════════════════════
# 4. 전처리 파이프라인 함수
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

    # Float32 수치 변환 (boolean 포함 — True→1.0 / False→0.0 자동 변환)
    num_cols = STATE_FEATURES + ACTION_FEATURES + ['Elapsed_Time', 'Step_Reward']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """2단계: 물리량을 AI가 이해하기 좋은 형태인 [-1, 1] 또는 [0, 1]로 압축"""
    df = df.copy()

    # ── 물리 상태 ─────────────────────────────────────────────────────────────
    # 속도: [-MAX_SPEED, MAX_SPEED] → [-1, 1]
    for col in ['Velocity_X', 'Velocity_Y', 'Velocity_Z']:
        df[col] = np.clip(df[col], -MAX_SPEED, MAX_SPEED) / MAX_SPEED

    # 자세각: [-180, 180] → [-1, 1]
    for col in ['Roll', 'Pitch', 'Yaw']:
        df[col] = df[col] / 180.0

    # ── 적 정보 ───────────────────────────────────────────────────────────────
    # 적 상대 좌표: [-MAX_ENEMY_DIST, MAX_ENEMY_DIST] → [-1, 1]
    for col in ENEMY_FEATURES:
        df[col] = np.clip(df[col], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST

    # 적 타입: {0, 1, 2, 3} → [0, 1]  (0=없음 / 1=사람 / 2=자폭 드론 / 3=재머)
    for col in ['Enemy1_Type', 'Enemy2_Type', 'Enemy3_Type']:
        df[col] = np.clip(df[col], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE

    # 위협 수준: [0, 100] → [0, 1]
    for col in ['Enemy1_Threat_Level', 'Enemy2_Threat_Level', 'Enemy3_Threat_Level']:
        df[col] = np.clip(df[col], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL

    # ── 라이다 ────────────────────────────────────────────────────────────────
    # 수평 24방향: [0, LIDAR_H_MAX] → [0, 1]
    for col in LIDAR_H_FEATURES:
        df[col] = np.clip(df[col], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX

    # 하향: [0, LIDAR_D_MAX] → [0, 1]
    df['Lidar_Down_90'] = np.clip(df['Lidar_Down_90'], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX

    # ── 목적지 ────────────────────────────────────────────────────────────────
    # 거리: [0, MAX_DEST_DIST] → [0, 1]
    df['Dest_Rel_Distance'] = (np.clip(df['Dest_Rel_Distance'], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST)

    # Yaw / Pitch 방향: 누적각 wrap → [-180, 180] → [-1, 1]
    for col in ['Dest_Rel_Yaw', 'Dest_Rel_Pitch']:
        df[col] = (((df[col] + 180.0) % 360.0) - 180.0) / 180.0

    # ── 맵 경계 거리 ──────────────────────────────────────────────────────────
    # X·Y 경계: [-MAX_BOUNDARY_XY, MAX_BOUNDARY_XY] → [-1, 1]
    for col in ['Dist_to_Max_X', 'Dist_to_Min_X', 'Dist_to_Max_Y', 'Dist_to_Min_Y']:
        df[col] = (np.clip(df[col], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY)

    # 천장: [0, MAX_CEILING] → [0, 1]
    df['Dist_to_Ceiling'] = (np.clip(df['Dist_to_Ceiling'], 0.0, MAX_CEILING) / MAX_CEILING)

    # ── 액션 및 보상 ──────────────────────────────────────────────────────────
    for col in ACTION_FEATURES:
        df[col] = np.clip(df[col], -1.0, 1.0)

    df['Step_Reward'] = np.clip(df['Step_Reward'], -REWARD_MAX, REWARD_MAX) / REWARD_MAX

    return df


def build_replay_buffer(df: pd.DataFrame) -> dict:
    """3단계: SAC 학습용 (상태, 행동, 보상, 다음상태, 종료) 전이 버퍼 조립"""
    s_list, a_list, r_list, s_next_list, term_list, trunc_list = [], [], [], [], [], []

    for _, group in df.groupby('Episode_Count'):
        group = group.sort_values('Step_Count').reset_index(drop=True)

        time_gap = group['Elapsed_Time'].diff().fillna(0.05)
        done_col = group['Done_State']

        # 정상 프레임 간격이거나, 게임 종료 스텝이면 통과
        valid = (((time_gap > 0) & (time_gap <= MAX_TIME_GAP)) | done_col.isin([1, 2, 3]))

        group = group[valid].reset_index(drop=True)
        if len(group) < 2:
            continue

        states  = group[STATE_FEATURES].values
        actions = group[ACTION_FEATURES].values
        rewards = group['Step_Reward'].values
        dones   = group['Done_State'].values
        steps   = group['Step_Count'].values

        for i in range(len(group) - 1):
            is_terminal_next = int(dones[i + 1]) in [1, 2, 3]

            # 연속되지 않은 스텝은 건너뜀 (종료 스텝 제외)
            if not is_terminal_next and steps[i + 1] != steps[i] + 1:
                continue

            # 보상은 s'(다음 상태) 도착 시점 기준으로 기록
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. 메인 실행부
# ══════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        print(f"[알림] '{INPUT_DIR}' 폴더에 변환할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 드론 비행 로그를 변환합니다.\n")

    for csv_path in sorted(csv_files):
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        npz_path = os.path.join(OUTPUT_DIR, f'{filename}.npz')

        print(f"[진행] {filename}.csv 변환 중...")

        df_clean = load_and_sanitize(csv_path)
        df_norm  = normalize(df_clean)
        buffer   = build_replay_buffer(df_norm)

        np.savez_compressed(npz_path, **buffer)

    print(f"\n[완료] 변환된 NPZ 파일들이 '{OUTPUT_DIR}'에 저장되었습니다.")
    print(f"       - State 차원: {len(STATE_FEATURES)}")
    print(f"       - Action 차원: {len(ACTION_FEATURES)}")


if __name__ == '__main__':
    main()