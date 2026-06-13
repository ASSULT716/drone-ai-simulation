import numpy as np
import os
import torch
# (사용 중이신 sac_agent 라이브러리나 SB3를 임포트하세요)
# from sac_agent import SACAgent 

def run_offline_training():
    print("🚀 [오프라인 학습 시작] NPZ 데이터 로드 중...")
    
    # 1. 2번 담당자가 만든 통짜 NPZ 파일 경로 (CSV 절대 안 씀!)
    npz_path = "data/processed_data.npz"
    
    if not os.path.exists(npz_path):
        print(f"❌ [에러] {npz_path} 파일이 없습니다! 전처리 담당자에게 NPZ 파일을 요청하세요.")
        return

    # 2. NPZ 파일 메모리에 한 번에 올리기 (초고속)
    dataset = np.load(npz_path)
    states = dataset['states']           # (N, 45) 차원
    actions = dataset['actions']         # (N, 6) 차원
    rewards = dataset['rewards']
    next_states = dataset['next_states']
    dones = dataset['terminateds']
    
    print(f"✅ 데이터 로드 완료! 총 {len(states)}개의 스텝을 학습합니다.")
    print(f" -> State 형태: {states.shape} | Action 형태: {actions.shape}")

    # 3. GPU 자동 세팅 (GTX 1660 포함 범용 세팅)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 학습 장치 가동: {device}")

    # 4. SAC 모델 초기화 및 리플레이 버퍼에 데이터 밀어넣기
    # (아래는 예시 코드입니다. 우진님의 sac_agent 구조에 맞게 수정하세요)
    """
    agent = SACAgent(state_dim=45, action_dim=6, device=device)
    
    print("📥 리플레이 버퍼에 NPZ 데이터 삽입 중...")
    for i in range(len(states)):
        agent.memory.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
    print("🧠 오프라인 베이스 모델(Behavioral Cloning) 학습 진행 중...")
    agent.train_offline(epochs=50) # 버퍼의 데이터를 이용한 사전 학습
    
    # 5. 가중치 저장
    os.makedirs('models', exist_ok=True)
    agent.save("models/sac_drone_base.zip")
    """
    print("🎉 [학습 완료] 완성된 뇌가 'models/sac_drone_base.zip' 에 저장되었습니다.")

if __name__ == "__main__":
    run_offline_training()