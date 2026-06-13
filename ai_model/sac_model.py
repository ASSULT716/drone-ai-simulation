import numpy as np
import gymnasium as gym

class DroneEngineAutoStartWrapper(gym.Wrapper):
    """
    [56차원 실시간 데이터 전처리 래퍼]
    전처리 담당자의 `STATE_FEATURES` 조립 순서와 정규화 수식을 100% 동일하게 반영
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(56,), dtype=np.float32)
        
        self.is_engine_on = False
        self.step_count = 0  

    def _preprocess_obs(self, raw_array):
        # 언리얼에서 받은 56개의 날것 데이터를 정규화하여 담을 빈 배열
        processed = np.zeros(56, dtype=np.float32)

        # 1. PHYSICS_FEATURES [0~2]: Velocity X, Y, Z (MAX_SPEED = 5500)
        processed[0:3] = np.clip(raw_array[0:3], -5500.0, 5500.0) / 5500.0

        # 2. PHYSICS_FEATURES [3~5]: Roll, Pitch, Yaw
        processed[3:6] = np.array(raw_array[3:6]) / 180.0

        # 3. ENEMY_FEATURES [6~14]: 적 1, 2, 3 상대 좌표 (MAX_ENEMY_DIST = 400000)
        processed[6:15] = np.clip(raw_array[6:15], -400000.0, 400000.0) / 400000.0

        # 4. ENEMY_META_FEATURES [15~20]: 적 타입(3)과 위협수준(100) 교차 배치
        processed[15] = np.clip(raw_array[15], 0.0, 3.0) / 3.0   # Enemy1_Type
        processed[16] = np.clip(raw_array[16], 0.0, 100.0) / 100.0 # Enemy1_Threat
        processed[17] = np.clip(raw_array[17], 0.0, 3.0) / 3.0   # Enemy2_Type
        processed[18] = np.clip(raw_array[18], 0.0, 100.0) / 100.0 # Enemy2_Threat
        processed[19] = np.clip(raw_array[19], 0.0, 3.0) / 3.0   # Enemy3_Type
        processed[20] = np.clip(raw_array[20], 0.0, 100.0) / 100.0 # Enemy3_Threat

        # 5. LIDAR_H_FEATURES [21~44]: 수평 라이다 24개 (LIDAR_H_MAX = 40000)
        processed[21:45] = np.clip(raw_array[21:45], 0.0, 40000.0) / 40000.0

        # 6. LIDAR_D_FEATURES [45]: 하향 라이다 1개 (LIDAR_D_MAX = 5000)
        processed[45] = np.clip(raw_array[45], 0.0, 5000.0) / 5000.0

        # 7. DEST_FEATURES [46~48]: 목적지 거리(200000), Yaw, Pitch (wrap 적용)
        processed[46] = np.clip(raw_array[46], 0.0, 200000.0) / 200000.0
        processed[47] = (((raw_array[47] + 180.0) % 360.0) - 180.0) / 180.0
        processed[48] = (((raw_array[48] + 180.0) % 360.0) - 180.0) / 180.0

        # 8. BOOL_FEATURES [49~50]: Is_On_Bool, Is_Ready_To_Fly_Bool
        processed[49:51] = raw_array[49:51]

        # 9. BOUNDARY_FEATURES [51~55]: Max_X, Min_X, Max_Y, Min_Y (125000), Ceiling (43000)
        processed[51:55] = np.clip(raw_array[51:55], -125000.0, 125000.0) / 125000.0
        processed[55] = np.clip(raw_array[55], 0.0, 43000.0) / 43000.0

        return processed

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        
        # 49번 인덱스가 Is_On_Bool
        self.is_engine_on = bool(initial_raw_state[49])
        self.step_count = 0  
        
        return self._preprocess_obs(initial_raw_state), info

    def step(self, action_from_ai):
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        self.step_count += 1
        cmd_engine = 0.0
        
        # 1.6초 엔진 자동 시동 개입 로직
        if not self.is_engine_on and self.step_count >= 32:
            cmd_engine = 1.0
            self.is_engine_on = True 
            
        final_action = np.array(action_list + [cmd_engine], dtype=np.float32)
        
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        
        # 업데이트된 엔진 상태 확인
        self.is_engine_on = bool(raw_next_state[49])
        
        return self._preprocess_obs(raw_next_state), reward, done, truncated, info