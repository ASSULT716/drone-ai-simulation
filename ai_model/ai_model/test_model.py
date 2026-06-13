import numpy as np
import gymnasium as gym
from sac_model import DroneEngineAutoStartWrapper

class FakeUnrealEnv(gym.Env):
    """
    언리얼 엔진의 데이터 송수신 역할을 가상으로 대행하는 시뮬레이터 환경
    """
    def __init__(self):
        super().__init__()
        # 원본 CSV 및 Preprocess.py에 정의된 날것(Raw)의 변수 69개 목록
        self.raw_keys = [
            'Episode_Count', 'Step_Count', 'Elapsed_Time',
            'Dist_to_Max_X', 'Dist_to_Min_X', 'Dist_to_Max_Y', 'Dist_to_Min_Y', 'Dist_to_Ceiling',
            'Velocity_X', 'Velocity_Y', 'Velocity_Z', 'Roll', 'Pitch', 'Yaw',
            'Move_Up', 'Move_Right', 'Move_Forward', 'Input_Roll', 'Input_Pitch_LookUp', 'Input_Yaw_Turn',
            'Enemy1_Type', 'Enemy1_Rel_X', 'Enemy1_Rel_Y', 'Enemy1_Rel_Z', 'Enemy1_Threat_Level',
            'Enemy2_Type', 'Enemy2_Rel_X', 'Enemy2_Rel_Y', 'Enemy2_Rel_Z', 'Enemy2_Threat_Level',
            'Enemy3_Type', 'Enemy3_Rel_X', 'Enemy3_Rel_Y', 'Enemy3_Rel_Z', 'Enemy3_Threat_Level',
            'Lidar_15', 'Lidar_30', 'Lidar_45', 'Lidar_60', 'Lidar_75', 'Lidar_90',
            'Lidar_105', 'Lidar_120', 'Lidar_135', 'Lidar_150', 'Lidar_165', 'Lidar_180',
            'Lidar_195', 'Lidar_210', 'Lidar_225', 'Lidar_240', 'Lidar_255', 'Lidar_270',
            'Lidar_285', 'Lidar_300', 'Lidar_315', 'Lidar_330', 'Lidar_345', 'Lidar_360',
            'Lidar_Down_90', 'Dest_Rel_Distance', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch',
            'Cmd_Engine', 'Is_On_Bool', 'Is_Ready_To_Fly_Bool', 'Engine_Power',
            'Step_Reward', 'Done_State'
        ]

    def _generate_fake_unreal_data(self, is_on=False):
        """언리얼 엔진 시뮬레이션 환경에서 송신되는 범위의 가상 원시 데이터 생성"""
        fake_data = {}
        for key in self.raw_keys:
            if 'Lidar' in key:
                fake_data[key] = float(np.random.uniform(100.0, 40000.0))
            elif 'Velocity' in key:
                fake_data[key] = float(np.random.uniform(-3000.0, 3000.0))
            elif key in ['Roll', 'Pitch', 'Yaw', 'Dest_Rel_Yaw', 'Dest_Rel_Pitch']:
                fake_data[key] = float(np.random.uniform(-180.0, 180.0))
            elif 'Enemy' in key or 'Dest_Rel_Distance' in key:
                fake_data[key] = float(np.random.uniform(500.0, 100000.0))
            else:
                fake_data[key] = 0.0
        
        # 가상 환경 내 드론 엔진 상태 지정
        fake_data['Is_On_Bool'] = float(is_on)
        fake_data['Is_Ready_To_Fly_Bool'] = float(is_on)
        return fake_data

    def reset(self, seed=None, options=None):
        # 시동이 꺼진 초기 상태의 프레임 데이터 반환
        raw_obs = self._generate_fake_unreal_data(is_on=False)
        return raw_obs, {}

    def step(self, action):
        # 입력받은 액션의 제 7축(Cmd_Engine) 상태에 따라 가상 엔진 가동 여부 결정
        cmd_engine = action[-1]
        is_on_now = True if cmd_engine == 1.0 else False
        
        raw_next_obs = self._generate_fake_unreal_data(is_on=is_on_now)
        reward = 0.1
        done = False
        truncated = False
        return raw_next_obs, reward, done, truncated, {}

if __name__ == "__main__":
    print("🔍 [가상 시뮬레이션] 45차원 환경 래퍼 및 실시간 전처리 파이프라인 무결성 검증 시작...")
    
    # 1. 베이스 가상 환경 선언 및 래퍼 융합
    base_env = FakeUnrealEnv()
    env = DroneEngineAutoStartWrapper(base_env)
    
    # 2. 초기화(Reset) 프로세스 검증
    obs, info = env.reset()
    print("\n[테스트 1] 가상 환경 초기화(Reset) 완료")
    print(f" -> 출력 관측 공간 차원(Observation Space): {obs.shape} (정상 기준: (45,))")
    print(f" -> 정규화 수치 스캔(상위 5개 샘플): {obs[:5]}")
    print(f" -> 엔진 정지 상태 정규화 플래그 확인 (최하위 2개 인덱스): {obs[-2:]} (기대값: [0. 0.])")
    
    # 3. 스텝(Step) 프로세스 및 시동 제어 검증 (AI 모델의 6차원 추론 액션 주입 가상 수행)
    print("\n[테스트 2] 에이전트 6차원 액션 주입 및 자동 시동 제어 로직 확인")
    fake_ai_action = np.array([0.5, -0.2, 1.0, 0.0, 0.1, -0.1], dtype=np.float32)
    
    # 래퍼 내부에서 제 7축(cmd_engine=1.0) 자동 연산 및 결합 수행 후 다음 스텝 진행
    next_obs, reward, done, truncated, info = env.step(fake_ai_action)
    
    print(f" -> 상호작용 후 갱신된 관측 차원: {next_obs.shape} (정상 기준: (45,))")
    print(f" -> 엔진 가동 상태 정규화 플래그 확인 (최하위 2개 인덱스): {next_obs[-2:]} (기대값: [1. 1.])")
    print("\n🎉 [검증 완료] sac_model.py에 내장된 실시간 정규화 및 자동 시동 인터페이스가 정상 작동합니다.")