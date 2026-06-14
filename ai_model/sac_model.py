import numpy as np
import gymnasium as gym

# 🌟 server_api.py와 완벽히 동일한 기준을 가져옵니다.
from config import (MAX_SPEED, MAX_STEPS, LIDAR_H_MAX, LIDAR_D_MAX, MAX_ENEMY_DIST, 
                    MAX_DEST_DIST, MAX_THREAT_LEVEL, MAX_ENEMY_TYPE, MAX_BOUNDARY_XY, 
                    MAX_CEILING, REWARD_MAX)

class DroneEngineAutoStartWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 🌟 핵심 수정: 62차원 State / 8차원 Action으로 확장 매핑 완료
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(62,), dtype=np.float32)
        
        self.current_unreal_step = 0
        self.has_pressed_i = False  

    def _preprocess_obs(self, raw_array):
        """
        🌟 server_api.py의 preprocess_live_obs와 100% 동일한 로직입니다.
        데이터셋의 62차원 Raw 데이터를 그대로 받아 일관되게 정규화합니다.
        """
        obs = np.array(raw_array, dtype=np.float32)

        # Index 0~2
        obs[0] = np.clip(obs[0], 0, MAX_STEPS) / MAX_STEPS
        obs[1] = np.clip(obs[1], 0, 3) / 3.0
        obs[2] = np.clip(obs[2], -REWARD_MAX, REWARD_MAX) / REWARD_MAX

        # Index 3~7
        obs[3:7] = np.clip(obs[3:7], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY
        obs[7] = np.clip(obs[7], 0.0, MAX_CEILING) / MAX_CEILING

        # Index 8~10 / 11~13
        obs[8:11] = np.clip(obs[8:11], -MAX_SPEED, MAX_SPEED) / MAX_SPEED
        obs[11:14] = obs[11:14] / 180.0

        # Index 14~17
        obs[14] = np.clip(obs[14], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX
        obs[15] = np.clip(obs[15], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST
        obs[16] = (((obs[16] + 180.0) % 360.0) - 180.0) / 180.0
        obs[17] = (((obs[17] + 180.0) % 360.0) - 180.0) / 180.0

        # Index 18~22 (Boolean 값은 스케일링 패스)

        # Index 23~37: 적 위협 정보 (3기 * 5차원)
        for i in range(3):
            idx = 23 + (i * 5)
            obs[idx] = np.clip(obs[idx], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
            obs[idx+1:idx+4] = np.clip(obs[idx+1:idx+4], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
            obs[idx+4] = np.clip(obs[idx+4], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL

        # Index 38~61: 수평 라이다 24방향
        obs[38:62] = np.clip(obs[38:62], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX

        return obs

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        self.current_unreal_step = int(initial_raw_state[0])
        self.has_pressed_i = False  
        return self._preprocess_obs(initial_raw_state), info

    def step(self, action_from_ai):
        # 🌟 AI 모델(SAC)이 출력한 8차원 액션을 수신
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        # 0.05초 주기에 맞춰 32스텝 기준으로 자동 시동 로직 세팅
        if self.current_unreal_step < 32:
            self.has_pressed_i = False

        pressing_i = 0.0
        if not self.has_pressed_i and self.current_unreal_step >= 32:
            pressing_i = 1.0
            self.has_pressed_i = True  
            
        # 🌟 Action Index 6 (Pressing_I) 강제 오버라이드 
        # (AI가 아무리 누르지 않으려 해도 시동은 강제로 걸어줌, 나머지 7개 차원은 AI 몫)
        action_list[6] = pressing_i
        
        final_action = np.array(action_list, dtype=np.float32)
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        
        self.current_unreal_step = int(raw_next_state[0])
        return self._preprocess_obs(raw_next_state), reward, done, truncated, info