import pandas as pd
import numpy as np

try:
    from ai_model.sac_agent import SACAgent
except ModuleNotFoundError:
    from sac_agent import SACAgent

def run_offline_training(csv_path):
    print(f"📂 CSV 데이터 로딩 중: {csv_path}")
    
    # 향후 엔진 담당자가 건네줄 CSV를 여기서 읽어들입니다.
    # df = pd.read_csv(csv_path)
    
    # 엔진 규격에 맞춘 임시 차원 설정
    state_dim = 45 
    action_dim = 5
    
    agent = SACAgent(state_dim, action_dim)
    print("🚀 오프라인 사전 학습(Offline Pre-training)을 시작합니다...")
    
    # 향후 df.iterrows() 로 루프를 돌며 버퍼에 넣고 agent.train_step() 실행
    
    agent.save_models()
    print("🎉 오프라인 학습 완료! 모델이 저장되었습니다.")

if __name__ == '__main__':
    # 테스트 실행 (실제 CSV 경로 연결 전)
    run_offline_training("data/raw_data.csv")