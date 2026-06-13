import numpy as np
import gymnasium as gym

class DroneEngineAutoStartWrapper(gym.Wrapper):
    """
    [실시간 데이터 전처리 래퍼]
    언리얼 수신 데이터(Dict)를 NPZ 데이터 스케일과 100% 동일하게 정규화하며,
    시작 1.6초 후에 드론 엔진 시동을 자동으로 제어합니다.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(45,), dtype=np.float32)
        
        # 시간 및 시동 제어를 위한 상태 변수 초기화
        self.is_engine_on = False
        self.episode_start_time = 0.0
        self.current_elapsed_time = 0.0

    def _preprocess_obs(self, raw_dict):
        # 1. Physics Features (속도 5500, 각도 180 기준 스케일링)
        vel_x = np.clip(raw_dict.get('Velocity_X', 0.0), -5500.0, 5500.0) / 5500.0
        vel_y = np.clip(raw_dict.get('Velocity_Y', 0.0), -5500.0, 5500.0) / 5500.0
        vel_z = np.clip(raw_dict.get('Velocity_Z', 0.0), -5500.0, 5500.0) / 5500.0
        roll = raw_dict.get('Roll', 0.0) / 180.0
        pitch = raw_dict.get('Pitch', 0.0) / 180.0
        yaw = raw_dict.get('Yaw', 0.0) / 180.0
        physics_arr = [vel_x, vel_y, vel_z, roll, pitch, yaw]

        # 2. Enemy Features (적 거리 400000 기준 스케일링)
        enemy_keys = ['Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z',
                      'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z',
                      'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z']
        enemy_arr = [np.clip(raw_dict.get(k, 0.0), -400000.0, 400000.0) / 400000.0 for k in enemy_keys]

        # 3. Lidar Features (라이다 40000 기준 스케일링)
        lidar_h_keys = [f'Lidar_{i * 15}' for i in range(1, 25)]
        lidar_h_arr = [np.clip(raw_dict.get(k, 0.0), 0.0, 40000.0) / 40000.0 for k in lidar_h_keys]
        lidar_d = np.clip(raw_dict.get('Lidar_Down_90', 0.0), 0.0, 5000.0) / 5000.0
        lidar_arr = lidar_h_arr + [lidar_d]

        # 4. Dest Features
        dest_dist = np.clip(raw_dict.get('Dest_Rel_Distance', 0.0), -200000.0, 200000.0) / 200000.0
        dest_yaw = (((raw_dict.get('Dest_Rel_Yaw', 0.0) + 180.0) % 360.0) - 180.0) / 180.0
        dest_pitch = (((raw_dict.get('Dest_Rel_Pitch', 0.0) + 180.0) % 360.0) - 180.0) / 180.0
        dest_arr = [dest_dist, dest_yaw, dest_pitch]

        # 5. Bool Features
        is_on = float(raw_dict.get('Is_On_Bool', 0.0))
        is_ready = float(raw_dict.get('Is_Ready_To_Fly_Bool', 0.0))
        bool_arr = [is_on, is_ready]

        # 최종 45차원 병합 반환
        return np.array(physics_arr + enemy_arr + lidar_arr + dest_arr + bool_arr, dtype=np.float32)

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        
        # 에피소드 시작 시점의 절대 시간 기록 및 현재 시간 동기화
        self.episode_start_time = float(initial_raw_state.get('Elapsed_Time', 0.0))
        self.current_elapsed_time = self.episode_start_time
        
        self.is_engine_on = bool(initial_raw_state.get('Is_On_Bool', False))
        return self._preprocess_obs(initial_raw_state), info

    def step(self, action_from_ai):
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        # 에피소드 시작 이후 흐른 시뮬레이션 시간 계산
        time_passed = self.current_elapsed_time - self.episode_start_time
        
        # 기본값은 시동 명령을 내리지 않음 (0.0)
        cmd_engine = 0.0
        
        # 아직 시동이 켜지지 않았고, 시뮬레이션 시간이 1.6초 이상 지났을 때만 시동 개입
        if not self.is_engine_on and time_passed >= 1.6:
            cmd_engine = 1.0
            self.is_engine_on = True 
            
        # 기존 6차원 액션에 제 7축 시동 명령어 플래그 결합
        final_action = np.array(action_list + [cmd_engine], dtype=np.float32)
        
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        
        # 다음 스텝 제어를 위해 시뮬레이션 시간 및 시동 상태 업데이트
        self.current_elapsed_time = float(raw_next_state.get('Elapsed_Time', 0.0))
        self.is_engine_on = bool(raw_next_state.get('Is_On_Bool', False))
        
        return self._preprocess_obs(raw_next_state), reward, done, truncated, info