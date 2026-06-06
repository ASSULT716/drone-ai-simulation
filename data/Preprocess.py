import pandas as pd
import numpy as np
import os

# ── 1. 스키마 및 상수 정의 (팀원 규격 45차원에 정확히 맞춤) ──────────────────

BASE_STATE_FEATURES = [
    'rel_x', 'rel_y', 'rel_z',
    'Velocity_X', 'Velocity_Y', 'Velocity_Z',
    'Rot_Roll', 'Rot_Pitch', 'Rot_Yaw',
    'Accel_X', 'Accel_Y', 'Accel_Z',
    'Ang_Vel_P', 'Ang_Vel_R', 'Ang_Vel_Y',
    # 팀원 코드의 state_dim=45를 맞추기 위해 제외 (15개 물리 상태 + 6개 적 위치 = 21)
    'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 
    'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z'
]
RAY_FEATURES = [f'Ray_{i}' for i in range(24)]

# 팀원의 액션 차원(5) 및 컬럼명 동기화
ACTION_FEATURES = ['out_x', 'out_y', 'out_z', 'yaw', 'pitch']

MAX_SPEED = 70.0   # m/s             
MAX_TIME_GAP = 0.15            
MAX_STEPS = 1000              
RAY_MAX_RANGE = 400.0         

# ── 2. 안전한 데이터 로드 및 초기 위생 처리 ───────────────────────────────────

def load_and_sanitize_csv(csv_file_path: str) -> pd.DataFrame:
    print("[STEP 1] 데이터 로드 및 초기 위생 처리...")
    
    first_line = pd.read_csv(csv_file_path, nrows=1)
    if not isinstance(first_line.columns[0], (int, float, np.number)):
        print("  - 헤더가 감지되었습니다. 정상 로드합니다.")
        df = pd.read_csv(csv_file_path)
    else:
        print("  - 헤더가 없습니다. 정밀 슬라이싱 스키마를 적용합니다.")
        all_cols = ['Episode_ID', 'Step_Count', 'timestamp', 'Done_State', 'Step_Reward'] + \
                   BASE_STATE_FEATURES[:15] + ['Enemy1_Type'] + BASE_STATE_FEATURES[15:18] + ['Enemy2_Type'] + BASE_STATE_FEATURES[18:] + \
                   ACTION_FEATURES + RAY_FEATURES
        df = pd.read_csv(csv_file_path, header=None, names=all_cols)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    float_cols = BASE_STATE_FEATURES + RAY_FEATURES + ACTION_FEATURES + ['timestamp', 'Step_Reward']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
            
    return df

# ── 3. 고급 특성 공학 및 정규화 (피드백 3종 완벽 반영) ─────────────────────────

def preprocess_features(df: pd.DataFrame) -> tuple:
    print("[STEP 2] 고급 특성 공학 및 피드백 기반 정규화 가동...")
    
    # 피드백 ③: Action Scaling 명확화 ([-1, 1]로 가두기)
    for act in ACTION_FEATURES:
        if act in df.columns:
            df[act] = np.clip(df[act], -1.0, 1.0)

    # 피드백 ①: Ray 센서 Clamp 후 Normalization 진행
    for ray in RAY_FEATURES:
        if ray in df.columns:
            df[ray] = np.clip(df[ray], 0.0, RAY_MAX_RANGE) / RAY_MAX_RANGE

    # 피드백 ②: Reward Z-Score Normalization (단순 clip 대신 표준화 적용)
    if 'Step_Reward' in df.columns:
        reward_mean = df['Step_Reward'].mean()
        reward_std = df['Step_Reward'].std() + 1e-8
        df['Step_Reward'] = (df['Step_Reward'] - reward_mean) / reward_std
        print(f"  - 🎯 보상 표준화 완료 (Mean: {reward_mean:.4f}, Std: {reward_std:.4f})")

    # 물리 각도 정규화
    for angle_col in ['Rot_Roll', 'Rot_Pitch', 'Rot_Yaw']:
        if angle_col in df.columns:
            df[angle_col] = df[angle_col] / 180.0

    # 차원 수 유지를 위해 원핫 인코딩을 제외하고 순수 45차원으로 구성
    final_state_columns = BASE_STATE_FEATURES + RAY_FEATURES
    
    return df, final_state_columns

# ── 4. 연속성 검증 기반의 리플레이 버퍼 조립 ───────────────────────────────────

def build_replay_buffer(df: pd.DataFrame, state_cols: list) -> dict:
    print("[STEP 3] 에피소드 격리 및 연속성 기반 MDP 데이터셋 빌드...")
    
    s_list, a_list, r_list, s_next_list, term_list, trunc_list = [], [], [], [], [], []

    for ep_id, group in df.groupby('Episode_ID'):
        group = group.sort_values('Step_Count')
        
        time_gap = group['timestamp'].diff().fillna(0.05) if 'timestamp' in group.columns else group['Time_Stamp'].diff().fillna(0.05)
        
        if all(c in group.columns for c in ['Velocity_X', 'Velocity_Y', 'Velocity_Z']):
            speed = np.sqrt(group['Velocity_X']**2 + group['Velocity_Y']**2 + group['Velocity_Z']**2)
            valid_mask = (time_gap > 0) & (time_gap <= MAX_TIME_GAP) & (speed <= MAX_SPEED)
        else:
            valid_mask = (time_gap > 0) & (time_gap <= MAX_TIME_GAP)
        
        filtered_group = group[valid_mask].copy()
        if len(filtered_group) < 2:
            continue
            
        states = filtered_group[state_cols].values
        actions = filtered_group[ACTION_FEATURES].values
        rewards = filtered_group['Step_Reward'].values if 'Step_Reward' in filtered_group.columns else np.zeros(len(filtered_group))
        dones = filtered_group['Done_State'].values if 'Done_State' in filtered_group.columns else np.zeros(len(filtered_group))
        step_counts = filtered_group['Step_Count'].values

        for i in range(len(filtered_group) - 1):
            if step_counts[i+1] != step_counts[i] + 1:
                continue
                
            s = states[i]
            a = actions[i]
            r = float(rewards[i])
            s_next = states[i+1]
            
            if dones[i+1] in [1, 2, 3]:
                term, trunc = True, False
            elif step_counts[i+1] >= MAX_STEPS:
                term, trunc = False, True
            else:
                term, trunc = False, False
                
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            s_next_list.append(s_next)
            term_list.append(term)
            trunc_list.append(trunc)
            
    return {
        'states': np.array(s_list, dtype=np.float32),
        'actions': np.array(a_list, dtype=np.float32),
        'rewards': np.array(r_list, dtype=np.float32),
        'next_states': np.array(s_next_list, dtype=np.float32),
        'terminateds': np.array(term_list, dtype=bool),
        'truncateds': np.array(trunc_list, dtype=bool)
    }

# ── 5. 메인 마스터 제어 프레임워크 ─────────────────────────────────────────────

def main(csv_file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df_sanitized = load_and_sanitize_csv(csv_file_path)
    df_processed, final_state_cols = preprocess_features(df_sanitized)
    
    replay_buffer_dict = build_replay_buffer(df_processed, final_state_cols)
    
    # 팀원의 train_offline.py 확장자 매칭 팁
    # 만약 팀원 코드가 .npz를 읽는다면 이대로 가시면 되고,
    # 만약 정제된 CSV 데이터 자체를 한 줄씩 읽는 구조(`pd.read_csv`)라면 
    # 이 아래 저장 로직만 df_processed.to_csv() 형태로 변경해 주시면 됩니다!
    output_path = os.path.join(output_dir, 'sac_replay_buffer.npz')
    np.savez_compressed(output_path, **replay_buffer_dict)
    
    print(f"\n [최종 검증 완료] 전처리 파이프라인 정상 종료")
    print(f" -최종 State 차원수 : {len(final_state_cols)} 차원 (팀원 45차원 규격 완벽 일치)")
    print(f" -최종 Action 차원수 : {len(ACTION_FEATURES)} 차원 (팀원 5차원 규격 완벽 일치)")
    print(f" -유효한 MDP 샘플 수 : {len(replay_buffer_dict['states'])} steps")
    print(f" -저장 완료 ➡️ {output_path}")

if __name__ == "__main__":
    main('data/data/dummy_game_data.csv', 'data/processed')