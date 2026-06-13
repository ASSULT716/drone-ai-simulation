import numpy as np
import gymnasium as gym

class DroneEngineAutoStartWrapper(gym.Wrapper):
    """
    [56차원 이진 데이터 전처리 래퍼]
    언리얼 수신 패킷(56개 Float 배열)을 NPZ 정규화 기준에 맞게 스케일링
    """
    def __init__(self, env):
        super().__init__(env)
        # 🌟 액션 6차원, 스테이트 56차원으로 완벽하게 분리 반영
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(56,), dtype=np.float32)
        
        self.is_engine_on = False
        self.step_count = 0  

    def _preprocess_obs(self, raw_array):
        # 56개짜리 빈 배열 생성
        processed = np.zeros(56, dtype=np.float32)

        # 1. [0~4] 가변형 경계 5개 (Max/Min X, Y, Ceiling) 
        processed[0:5] = np.clip(raw_array[0:5], -100000.0, 100000.0) / 100000.0

        # 2. [5~7] 드론 속도 3개 (Velocity X, Y, Z)
        processed[5:8] = np.clip(raw_array[5:8], -5500.0, 5500.0) / 5500.0

        # 3. [8~10] 드론 자세 3개 (Roll, Pitch, Yaw)
        processed[8:11] = np.array(raw_array[8:11]) / 180.0

        # (※ 15~20번 조종 입력값 6개는 Action이므로 State에서 제외됨 -> 완벽한 싱크로율!)

        # 4. [11~25] 적 탐지 15개 (3기 × 5개 피처)
        for i in range(3):
            base_idx = 11 + (i * 5)
            processed[base_idx] = raw_array[base_idx] / 3.0  # 적 타입 (1,2,3)
            processed[base_idx+1:base_idx+4] = np.clip(raw_array[base_idx+1:base_idx+4], -400000.0, 400000.0) / 400000.0
            processed[base_idx+4] = raw_array[base_idx+4] / 100.0 # 위협 수준

        # 5. [26~49] 정면 라이다 센서 24개 (15도 ~ 360도)
        processed[26:50] = np.clip(raw_array[26:50], 0.0, 40000.0) / 40000.0

        # 6. [50] 하향 라이다 센서 1개
        processed[50] = np.clip(raw_array[50], 0.0, 5000.0) / 5000.0

        # 7. [51~53] 목적지 정보 3개 (거리, Yaw, Pitch)
        processed[51] = np.clip(raw_array[51], -200000.0, 200000.0) / 200000.0
        processed[52:54] = np.array(raw_array[52:54]) / 180.0

        # 8. [54~55] 엔진 상태 부울 2개 (Is_On, Is_Ready)
        processed[54:56] = raw_array[54:56]

        # 최종 56차원 배열 리턴!
        return processed

    def reset(self, **kwargs):
        initial_raw_state, info = self.env.reset(**kwargs)
        
        # 54번 인덱스가 Is_On_Bool 값
        self.is_engine_on = bool(initial_raw_state[54])
        self.step_count = 0  
        
        return self._preprocess_obs(initial_raw_state), info

    def step(self, action_from_ai):
        action_list = action_from_ai.tolist() if isinstance(action_from_ai, np.ndarray) else list(action_from_ai)
        
        self.step_count += 1
        cmd_engine = 0.0
        
        # 0.05초 * 32 프레임 = 1.6초 엔진 자동 시동
        if not self.is_engine_on and self.step_count >= 32:
            cmd_engine = 1.0
            self.is_engine_on = True 
            
        # 6차원 액션 + 1차원 엔진 시동 = 7차원 전송 
        final_action = np.array(action_list + [cmd_engine], dtype=np.float32)
        
        raw_next_state, reward, done, truncated, info = self.env.step(final_action)
        self.is_engine_on = bool(raw_next_state[54])
        
        return self._preprocess_obs(raw_next_state), reward, done, truncated, info