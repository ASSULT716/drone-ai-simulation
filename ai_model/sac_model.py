import numpy as np
import gymnasium as gym

class DroneEngineAutoStartWrapper(gym.Wrapper):
    """
    [AI 전용 환경 래퍼 & 실시간 데이터 전처리]
    1. 엔진 시동 강제 제어 (7번째 인덱스 개입)
    2. 언리얼 수신 데이터(Raw)를 NPZ 생성 기준과 100% 동일하게 실시간 스케일링
    """
    def __init__(self, env):
        super().__init__(env)
        # Action Space (6차원)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # Observation Space (45차원)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(45,), dtype=np.float32)
        self.is_engine_on = False

    def _preprocess_obs(self, raw_dict):
        """Preprocess.py의 normalize() 함수와 수학적으로 완벽히 동일한 실시간 전처리 함수"""
        
        # 1. Physics Features
        vel_x = np.clip(raw_dict.get('Velocity_X', 0.0), -5500.0, 5500.0) / 5500.0
        vel_y = np.clip(raw_dict.get('Velocity_Y', 0.0), -5500.0, 5500.0) / 5500.0
        vel_z = np.clip(raw_dict.get('Velocity_Z', 0.0), -5500.0, 5500.0) / 5500.0
        roll = raw_dict.get('Roll', 0.0) / 180.0
        pitch = raw_dict.get('Pitch', 0.0) / 180.0
        yaw = raw_dict.get('Yaw', 0.0) / 180.0
        physics_arr = [vel_x, vel_y, vel_z, roll, pitch, yaw]

        # 2. Enemy Features
        enemy_keys = ['Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z',
                      'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z',
                      'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z']
        enemy_arr = [np.clip(raw_dict.get(k, 0.0), -400000.0, 400000.0) / 400000.0 for k in enemy_keys]

        # 3. Lidar Features
        lidar_h_keys = [f'Lidar_{i * 15}' for i in range(1, 25)]
        lidar_h_arr = [np.clip(raw_dict.get(k, 0.0), 0.0, 40000.0) / 40000.0 for k in lidar_h_keys]
        lidar_d = np.clip(raw_dict.get('Lidar_Down_90', 0.0), 0.0, 5000.0) / 5000.0
        lidar_arr = lidar_h_arr + [lidar_d]

        # 4. Dest Features
        dest_dist = np.clip(raw_dict.get('Dest_Rel_Distance', 0.0), -200000.0, 200000.0) / 200000.0
        dest_yaw = (((raw_dict.get('Dest_Rel_Yaw', 0.0) + 180.0) % 360.0) - 180.0) / 180.0
        dest_pitch = (((raw_dict.get('Dest_Rel_Pitch', 0.0) + 180.0) % 360.0) - 180.0) / 180.0
        dest_arr = [dest_dist, dest_yaw, dest_pitch]

        # 5. Bool Features (True/False -> 1.0/0.0)
        is_on = float(raw_dict.get('Is_On_Bool', 0.0))
        is_ready = float(raw_dict.get('Is_Ready_To_Fly_Bool', 0.0))
        bool_arr = [is_on, is_ready]

        # 최종 45차원 데이터 결합
        state_vector = np.array(physics_arr + enemy_arr + lidar_arr + dest_arr + bool_arr, dtype=np.float32)
        return state_vector

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        
        # 엔진 상태 체크 (bool 피처의 첫 번째 인덱스)
        self.is_engine_on = bool(initial_raw_state.get('Is_On_Bool', False))
        
        # 날것의 데이터를 전처리해서 모델에 전달
        processed_state = self._preprocess_obs(initial_raw_state)
        return processed_state, info

    def step(self, action_from_ai):
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        if not self.is_engine_on:
            cmd_engine = 1.0 
            self.is_engine_on = True 
            print("[Wrapper] 엔진 시동(cmd_engine=1.0) 자동 주입 완료.")
        else:
            cmd_engine = 0.0 
            
        final_action = np.array(action_list + [cmd_engine], dtype=np.float32)
        
        # 1. 언리얼 엔진에 7차원 액션(비행6+엔진1) 전송 후 다음 상태 수신
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        
        # 2. 엔진 켜짐 여부 업데이트
        self.is_engine_on = bool(raw_next_state.get('Is_On_Bool', False))
        
        # 3. 모델이 이해할 수 있도록 날것의 데이터를 NPZ와 동일한 기준으로 정규화
        processed_state = self._preprocess_obs(raw_next_state)
        
        return processed_state, reward, done, truncated, info