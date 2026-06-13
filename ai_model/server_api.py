from flask import Flask, request, jsonify
import numpy as np
import torch
# from sac_agent import SACAgent
# from sac_model import DroneEngineAutoStartWrapper (사용하는 환경 구성에 따라 맞게 임포트)

app = Flask(__name__)

# 1. 기지국 가동 시, 오프라인 학습으로 완성된 뼈대 모델(.zip) 단 한 번 로드
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# agent = SACAgent(device=device)
# agent.load_models("models/sac_drone_base.zip")
print("✅ [서버 준비] 베이스 모델 로드 완료. 실시간 무전 대기 중...")

# 이전 스텝의 데이터를 기억하기 위한 전역 변수
previous_state = None

@app.route('/predict', methods=['POST'])
def predict():
    global previous_state
    
    # 2. 언리얼 엔진에서 0.05초마다 쏘는 JSON 날것의 데이터 수신
    raw_dict = request.json
    
    if not raw_dict:
        return jsonify({"error": "No data received"}), 400

    # 3. 우진님의 래퍼 전처리 로직을 이용해 45차원으로 깎아냄 (임시로 래퍼 함수 직접 호출하는 방식 예시)
    # 실제로는 env.step() 구조에 맞게 조립되어 있을 것입니다.
    # current_state = env._preprocess_obs(raw_dict) 
    
    # (아래는 임시 더미 데이터입니다. 실제 래퍼를 통과한 current_state를 사용하세요)
    current_state = np.zeros(45, dtype=np.float32) 
    done = bool(raw_dict.get('Done_State', 0) in [1, 2, 3])
    reward = float(raw_dict.get('Step_Reward', 0.1))

    # 4. (선택 사항) 실시간 파인 튜닝 로직
    """
    if previous_state is not None:
        agent.store_transition(previous_state, temp_action, reward, current_state, done)
        agent.train_step() # 실시간 1스텝 학습
    """

    # 5. AI의 뇌에서 액션(6차원) 도출
    # action = agent.select_action(current_state)
    action = np.zeros(6, dtype=np.float32) # 임시 더미 액션

    if done:
        previous_state = None
        print("🚨 드론 리스폰! 에피소드 초기화")
    else:
        previous_state = current_state

    # 6. 언리얼 엔진으로 답장(Response) 쏘기
    return jsonify({"action": action.tolist()})

if __name__ == '__main__':
    # 서버 가동
    app.run(host='0.0.0.0', port=5000, debug=False)