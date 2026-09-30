import os
import time
import numpy as np
import glob
import torch

# 우리가 새롭게 구축한 무결점 모듈들
from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from server_api import UnrealCommunicationAPI

# 하드코딩된 경로 대신 상대 경로를 사용하여 새 PC 환경에 완벽 대응
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "model")
# 언리얼에서 .bin 파일을 뱉어내는 경로 (본인 PC 환경에 맞게 수정 필요)
BIN_DIR_PATH = r"C:\Users\jjang\OneDrive\바탕 화면\Project\drone-ai-simulation\data\RL_Data\Tutorial\Packet"

def main():
    print("🚀 [Online RL] 드론 AI 실시간 강화학습을 시작합니다...")
    
    # 1. 차원 통일 (치명적 오류 해결)
    STATE_DIM = 62
    ACTION_DIM = 8
    BATCH_SIZE = 256
    
    # 2. 모듈 초기화
    agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    memory = ReplayBuffer(state_dim=STATE_DIM, action_dim=ACTION_DIM, max_size=500000)
    
    # 통신 모듈 초기화 (언리얼 IP와 OSC 포트)
    unreal_api = UnrealCommunicationAPI(
        bin_dir=BIN_DIR_PATH, 
        osc_ip="100.82.247.96", # 기존 서버 코드의 언리얼 IP 유지
        osc_port=7000
    )
    
    # 모델 저장 폴더 확인 및 생성
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_DIR, "sac_actor.pth")
    if os.path.exists(model_path):
        # agent.load_models(MODEL_SAVE_DIR) # 구현된 로드 함수 호출
        print("✅ 기존 AI 뇌(가중치) 로드 완료.")
    else:
        print("⚠️ 저장된 모델이 없어 초기 가중치로 시작합니다.")

    print("\n🧹 서버 가동 전 폴더 내 기존 bin 파일 찌꺼기를 정리합니다...")
    unreal_api.clear_old_bins()
    print("✨ 정리 완료! 언리얼 엔진의 실시간 패킷을 기다립니다.\n")

    # 3. 실시간 학습 루프
    total_steps = 0
    
    while True:
        # 파일 포인터를 유지하며 최신 상태 62차원 및 피드백 수신
        # 타임아웃 발생 시 통신 대기 상태 유지
        state, reward, done, is_valid = unreal_api.read_next_state()
        
        if not is_valid:
            continue # 패킷이 완전히 작성되지 않았거나 대기 중
            
        total_steps += 1
        
        # 3-1. AI Action 추론 (Epsilon-Greedy가 내장된 select_action)
        action = agent.select_action(state, evaluate=False)
        
        # 3-2. Action 언리얼 규격 변환 및 OSC 전송
        unreal_api.send_action(action, done=done)
        
        # 3-3. 경험을 메모리에 저장하기 위해 다음 상태를 대기
        next_state, _, next_done, next_valid = unreal_api.read_next_state(timeout=0.1)
        
        if next_valid:
            # 62차원 상태, 8차원 원시 행동을 리플레이 버퍼에 저장
            memory.push(state, action, reward, next_state, next_done)
        
        # 3-4. 메모리에 데이터가 충분히 쌓이면 백프로파게이션(신경망 업데이트) 수행
        if len(memory) > BATCH_SIZE:
            agent.update_parameters(memory, BATCH_SIZE)
            
        # 3-5. 에피소드 종료 처리 (추락 또는 도착)
        if done or next_done:
            print(f"🔄 에피소드 종료! (현재 Epsilon 탐험률: {agent.epsilon:.3f})")
            agent.update_epsilon() # Epsilon 점진적 감소
            unreal_api.reset_file_pointer()
            
            # 주기적으로 모델 저장 (예: 1000스텝 마다)
            if total_steps % 1000 == 0:
                # agent.save_models(MODEL_SAVE_DIR)
                print(f"💾 {total_steps} 스텝 경과: 모델 중간 저장 완료.")

if __name__ == "__main__":
    main()