import pandas as pd
import numpy as np
import os
import glob

# ==============================================================================
# 1. 경로 및 설정
# ==============================================================================
INPUT_DIR  = 'data\\raw_data'
OUTPUT_DIR = 'data\\processed_data'

# [정밀 동기화] 언리얼 엔진에서 생성되는 총 72개 원본 CSV 컬럼의 정확한 순서
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
    'Step_Reward', 'Done_State', 'Pressing_I', 'Is_AI_Controlled', 'Is_Pressing_Backspace'
]

# ==============================================================================
# 2. .bin 파일 규격과 100% 일치시킨 State(62차원) 및 Action(6차원) 정의
# ==============================================================================
STATE_FEATURES = [
    'Current_step',          # 1. Current_step (스텝 번호)
    'Done_State',            # 2. Done_state ({0:진행중, 1:성공, 2:피격, 3:이탈})
    'Step_Reward',           # 3. Step_reward (보상 변수)
    'Dist_to_Max_X',         # 4. Distance to Max X
    'Dist_to_Min_X',         # 5. Distance to Min X
    'Dist_to_Max_Y',         # 6. Distance to Max Y
    'Dist_to_Min_Y',         # 7. Distance to Min Y
    'Dist_to_Ceiling',       # 8. Distance to ceiling
    'Velocity_X',            # 9. Velocity X
    'Velocity_Y',            # 10. Velocity Y
    'Velocity_Z',            # 11. Velocity Z
    'Roll',                  # 12. Roll 속도
    'Pitch',                 # 13. Pitch 속도
    'Yaw',                   # 14. Yaw 속도
    'Lidar_Down_90',         # 15. 하향 90도 라이다 탐지거리 (지면 절대거리)
    'Dest_Rel_Distance',     # 16. 목적지 상대적 거리
    'Dest_Rel_Yaw',          # 17. 목적지 상대적 yaw 방향
    'Dest_Rel_Pitch',        # 18. 목적지 상대적 pitch 방향
    'Is_On_Bool',            # 19. 엔진 전원 여부
    'Is_Ready_To_Fly_Bool',  # 20. 엔진 추력 100% 여부
    'Pressing_I',            # 21. I키 입력 여부
    'Is_AI_Controlled',      # 22. AI 조종 여부 (1=AI, 0=사람)
    'Is_Pressing_Backspace', # 23. Backspace 키 입력 여부
    # 24~28. 가장 높은 위협 수준의 적 정보 (Type, Rel_X, Rel_Y, Rel_Z, Threat)
    'Enemy1_Type', 'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 'Enemy1_Threat_Level',
    # 29~33. 두 번째로 높은 위협 수준의 적 정보
    'Enemy2_Type', 'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z', 'Enemy2_Threat_Level',
    # 34~38. 세 번째로 높은 위협 수준의 적 정보
    'Enemy3_Type', 'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z', 'Enemy3_Threat_Level',
] + [f'Lidar_{i * 15}' for i in range(1, 25)]  # 39~62. 정면 기준 15도~360도 라이다 (24차원)

# 조종 입력 Action (6차원)
ACTION_FEATURES = [
    'Move_Up', 'Move_Right', 'Move_Forward',
    'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn'
]

# ==============================================================================
# 3. 데이터 필터링용 상숫값
# ==============================================================================
MAX_TIME_GAP = 0.15      # 초    | 프레임 드랍 판단 기준
MAX_STEPS    = 10000      # 스텝  | 에피소드 최대 길이

# ==============================================================================
# 4. 전처리 파이프라인 함수 (스케일링 완전 제거 버전)
# ==============================================================================
def load_and_sanitize(csv_path: str) -> pd.DataFrame:
    """1단계: CSV 로드, 데이터 유효성 검사 및 float32 형변환 (스케일링 없음)"""
    df = pd.read_csv(csv_path, header=None, names=ALL_COLS)

    # 초기화 순간 발생하는 엔진 오작동 버그 제어 로직 (기존 유지)
    if (df['Cmd_Engine'] == 1).any():
        first_one_idx = (df['Cmd_Engine'] == 1).idxmax()
        mask = (df.index < first_one_idx) & (df['Cmd_Engine'] == 2)
        if mask.sum() > 0:
            df.loc[mask, 'Cmd_Engine'] = 0

    # 무한대(inf)값 제거 및 결측치 드랍
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 모든 분석 대상 컬럼들을 float32 데이터 타입으로 표준화 (메모리 최적화 및 AI 인풋 규격용)
    target_cols = list(set(STATE_FEATURES + ACTION_FEATURES + ['Episode_Count', 'Elapsed_Time']))
    for col in target_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

    return df


def build_replay_buffer(df: pd.DataFrame) -> dict:
    """2단계: 데이터셋을 순수한 원본 상태의 SAC 학습용 전이(Transition) 버퍼로 결합"""
    s_list, a_list, r_list, s_next_list, term_list, trunc_list = [], [], [], [], [], []

    # 에피소드 단위로 그룹화하여 데이터의 연속성 보장
    for _, group in df.groupby('Episode_Count'):
        group = group.sort_values('Current_step').reset_index(drop=True)

        time_gap = group['Elapsed_Time'].diff().fillna(0.05)
        done_col = group['Done_State']

        # 정상적인 타임스텝이거나 에피소드가 정상 종료되는 시점 필터링
        valid = (((time_gap > 0) & (time_gap <= MAX_TIME_GAP)) | done_col.isin([1, 2, 3]))
        group = group[valid].reset_index(drop=True)
        
        if len(group) < 2:
            continue

        # .bin의 배열 순서와 완벽히 일치하는 원본 데이터 추출
        states  = group[STATE_FEATURES].values
        actions = group[ACTION_FEATURES].values
        rewards = group['Step_Reward'].values
        dones   = group['Done_State'].values
        steps   = group['Current_step'].values

        for i in range(len(group) - 1):
            is_terminal_next = int(dones[i + 1]) in [1, 2, 3]

            # 프레임이 중간에 끊긴 부적절한 전이는 수집 제외
            if not is_terminal_next and steps[i + 1] != steps[i] + 1:
                continue

            # 강화학습 정의에 의거, 다음 스텝(i+1)에서 획득한 보상 매칭
            r = float(rewards[i + 1])

            # 환경 종료(Termination) 및 제한시간 초과(Truncation) 플래그 설정
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


# ==============================================================================
# 5. 메인 실행부
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

    if not csv_files:
        print(f"[알림] '{INPUT_DIR}' 폴더에 변환할 CSV 파일이 없습니다.")
        return

    print(f"총 {len(csv_files)}개의 드론 비행 로그를 정밀 동기화(Raw 데이터) 모드로 변환합니다.\n")

    for csv_path in sorted(csv_files):
        filename = os.path.splitext(os.path.basename(csv_path))[0]
        npz_path = os.path.join(OUTPUT_DIR, f'{filename}.npz')

        print(f"[진행] {filename}.csv 변환 중...")

        df_clean = load_and_sanitize(csv_path)
        buffer   = build_replay_buffer(df_clean)  # 정규화 함수(normalize)를 거치지 않고 바로 버퍼 생성

        np.savez_compressed(npz_path, **buffer)

    print(f"\n[완료] 변환이 완벽히 끝났습니다. 결과 파일이 '{OUTPUT_DIR}'에 저장되었습니다.")
    print(f"    - 데이터는 스케일링이 없는 'Raw 데이터 상태'입니다.")
    print(f"    - 동기화된 State 차원 수: {len(STATE_FEATURES)}차원 (.bin 파일과 100% 동일)")
    print(f"    - 동기화된 Action 차원 수: {len(ACTION_FEATURES)}차원")


if __name__ == '__main__':
    main()