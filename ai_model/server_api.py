from flask import Flask, request, jsonify
import numpy as np

try:
    from ai_model.sac_agent import SACAgent
except ModuleNotFoundError:
    from sac_agent import SACAgent

app = Flask(__name__)

agent = None
previous_state = None
EXPECTED_ACTION_DIM = 5 

@app.route('/predict', methods=['POST'])
def predict():
    global agent, previous_state
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "데이터가 없습니다"}), 400
        
    current_state = data.get("state")
    reward = data.get("reward", 0.0)
    done = data.get("done", False)
    
    state_dim = len(current_state)
    
    # 🌟 엔진에서 받은 데이터에 맞춰 뇌를 조립하고 오프라인 가중치 이식
    if agent is None or agent.state_dim != state_dim:
        print(f"\n📡 [입력 규격 감지] 센서 및 상태 데이터: {state_dim}차원")
        agent = SACAgent(state_dim=state_dim, action_dim=EXPECTED_ACTION_DIM)
        agent.load_models() 
        previous_state = current_state

    # 1. 온라인 상호작용 데이터 버퍼에 저장
    temp_action = np.zeros(EXPECTED_ACTION_DIM) 
    agent.store_transition(previous_state, temp_action, reward, current_state, done)
    
    # 2. 실시간 파인튜닝 진행
    agent.train_step()

    # 3. 새로운 액션 도출
    action = agent.select_action(current_state)
    
    if done:
        previous_state = None
        print("🚨 드론 리스폰! 에피소드 초기화")
    else:
        previous_state = current_state

    return jsonify({"action": action.tolist()})

if __name__ == '__main__':
    print("🚀 [하이브리드 SAC 통신 서버] 가동 완료! (포트: 5000)")
    app.run(host='0.0.0.0', port=5000)