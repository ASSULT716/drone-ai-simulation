import os
import glob
import numpy as np

# 🌟 같은 폴더(ai_model)에 있는 에이전트 클래스 참조
try:
    from sac_agent import SACAgent
except ModuleNotFoundError:
    from ai_model.sac_agent import SACAgent

# 🌟 파일 위치가 ai_model 내부이므로, 상위 프로젝트 루트 경로를 자동 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed_data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "ai_model", "model")

def load_npz_data(filepath):
    data = np.load(filepath)
    return {
        'states': data['states'],
        'actions': data['actions'],
        'rewards': data['rewards'],
        'next_states': data['next_states'],
        'terminateds': data['terminateds'],
        'truncateds': data['truncateds']
    }

def train_offline():
    # processed_data 폴더 안의 모든 npz 파일을 탐색
    npz_files = glob.glob(os.path.join(DATA_DIR, "*.npz"))
    
    if not npz_files:
        print(f"🚨 '{DATA_DIR}' 폴더에 학습할 데이터(.npz)가 없습니다. 전처리를 먼저 진행해주세요.")
        return

    print(f"🚀 [오프라인 학습 시작] 총 {len(npz_files)}개의 데이터 파일을 로드합니다.")

    # 🌟 62차원 State, 8차원 Action 완벽 동기화하여 에이전트 선언
    agent = SACAgent(state_dim=62, action_dim=8)
    
    # 기존 가중치(.pth)가 있다면 누적해서 디벨롭 학습 진행
    agent.load_models(path=MODEL_SAVE_DIR + "/")

    total_transitions_added = 0
    
    # 모든 npz 파일의 데이터를 추출하여 에이전트 메모리에 적재
    for npz_file in npz_files:
        print(f"📥 데이터 적재 중... [{os.path.basename(npz_file)}]")
        batch_data = load_npz_data(npz_file)
        
        num_samples = len(batch_data['states'])
        for i in range(num_samples):
            state = batch_data['states'][i]
            action = batch_data['actions'][i]
            reward = float(batch_data['rewards'][i])
            next_state = batch_data['next_states'][i]
            
            # terminal(성공/죽음) 플래그 통합
            done = bool(batch_data['terminateds'][i] or batch_data['truncateds'][i])
            
            agent.store_transition(state, action, reward, next_state, done)
            total_transitions_added += 1

    print(f"\n✅ 총 {total_transitions_added}개의 경험(Transition)이 메모리에 정상 적재되었습니다.")
    print("🧠 딥러닝 역전파(Backpropagation) 최적화 훈련을 시작합니다...")

    # 적재된 데이터 양에 비례하여 훈련 횟수 설정 (데이터 10개당 1회 학습 비율)
    train_iterations = total_transitions_added // 10 
    
    for iteration in range(train_iterations):
        agent.train_step()
        if (iteration + 1) % 1000 == 0:
            print(f"📊 학습 진행률: {iteration + 1} / {train_iterations} 완료")

    # 새롭게 업데이트된 최적의 뇌 가중치 파일 덮어쓰기 저장
    agent.save_models(path=MODEL_SAVE_DIR + "/")
    print("\n🎉 [학습 완료] 완성된 뇌 가중치가 'ai_model/model/' 폴더에 안전하게 저장되었습니다!")

if __name__ == "__main__":
    train_offline()