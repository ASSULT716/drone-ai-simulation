import numpy as np
import gymnasium as gym

class DroneEngineAutoStartWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 🌟 버그 수정: AI 모델 자체의 규격은 데이터셋(.npz)과 동일하게 '6차원'이어야 충돌이 안 납니다!
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(56,), dtype=np.float32)
        
        self.current_unreal_step = 0
        self.has_pressed_i = False  

    def _preprocess_obs(self, raw_array):
        processed = np.zeros(56, dtype=np.float32)
        
        # 60차원 수신 데이터 기준 정밀 인덱싱 매핑
        processed[0:4] = np.clip(raw_array[4:8], -125000.0, 125000.0) / 125000.0
        processed[4] = np.clip(raw_array[8], 0.0, 43000.0) / 43000.0
        processed[5:8] = np.clip(raw_array[9:12], -5500.0, 5500.0) / 5500.0
        processed[8:11] = np.array(raw_array[12:15]) / 180.0
        processed[11] = np.clip(raw_array[15], 0.0, 5000.0) / 5000.0
        processed[12] = np.clip(raw_array[16], 0.0, 200000.0) / 200000.0
        processed[13] = (((raw_array[17] + 180.0) % 360.0) - 180.0) / 180.0
        processed[14] = (((raw_array[18] + 180.0) % 360.0) - 180.0) / 180.0
        processed[15:17] = raw_array[19:21]
        
        for i in range(3):
            base_raw = 21 + (i * 5)
            base_proc = 17 + (i * 5)
            processed[base_proc] = np.clip(raw_array[base_raw], 0.0, 3.0) / 3.0
            processed[base_proc+1:base_proc+4] = np.clip(raw_array[base_raw+1:base_raw+4], -400000.0, 400000.0) / 400000.0
            processed[base_proc+4] = np.clip(raw_array[base_raw+4], 0.0, 100.0) / 100.0
            
        processed[32:56] = np.clip(raw_array[36:60], 0.0, 40000.0) / 40000.0
        return processed

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        self.current_unreal_step = int(initial_raw_state[0])
        self.has_pressed_i = False  
        return self._preprocess_obs(initial_raw_state), info

    def step(self, action_from_ai):
        # AI 모델(SAC)이 출력한 깨끗한 6차원 액션 수신
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        # 🌟 0.05초 주기에 맞춰 32스텝 기준으로 자동 시동 로직 세팅
        if self.current_unreal_step < 32:
            self.has_pressed_i = False

        pressing_i = 0.0
        if not self.has_pressed_i and self.current_unreal_step >= 32:
            pressing_i = 1.0
            self.has_pressed_i = True  
            
        # 🌟 래퍼 내부에서 6축 조종값 뒤에 시동축을 붙여 최종 7차원으로 환경에 전송합니다.
        final_action = np.array(action_list + [pressing_i], dtype=np.float32)
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        
        self.current_unreal_step = int(raw_next_state[0])
        return self._preprocess_obs(raw_next_state), reward, done, truncated, info