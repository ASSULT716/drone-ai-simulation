import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import os

# 1. 아까 학습할 때 썼던 AI 뇌(신경망) 구조를 그대로 가져옵니다.
class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x)

# 2. 통신소(Flask 서버) 만들기
app = Flask(__name__)

# 3. 서버가 켜질 때 우리가 완성한 AI 뇌(bc_model.pth) 장착하기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BCNet().to(device)

MODEL_PATH = 'model/bc_model.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval() # 🌟 학습 모드 종료! 이제부터는 실전(추론) 모드로 전환
    print(f"✅ 똑똑해진 AI 뇌 로딩 완료! (가동 장치: {device})")
else:
    print(f"❌ 에러: {MODEL_PATH} 파일을 찾을 수 없습니다. 학습을 먼저 진행해주세요!")

# 4. 언리얼 엔진이 접근할 창구(API 엔드포인트) 열어주기
@app.route('/predict', methods=['POST'])
def predict_action():
    try:
        # 언리얼 엔진이 보내준 드론의 현재 상태(입력 6개) 받기
        data = request.json
        
        input_state = [
            float(data['rel_x']),
            float(data['rel_y']),
            float(data['rel_z']),
            float(data['threat']),
            float(data['hp']),
            float(data['is_damaged'])
        ]
        
        # 데이터를 파이토치 형태(Tensor)로 바꿔서 GPU로 전송
        input_tensor = torch.tensor([input_state], dtype=torch.float32).to(device)
        
        # 🌟 AI 뇌에게 "어떻게 피할까?" 물어보고 정답(조작 5개) 얻기
        with torch.no_grad(): # 추론할 때는 컴퓨터 자원을 아끼기 위해 no_grad() 사용
            output_tensor = model(input_tensor)
        
        action = output_tensor[0].cpu().numpy()
        
        # 언리얼 엔진에게 조작 명령(출력 5개)을 예쁘게 포장해서 돌려보내기
        response = {
            "out_x": float(action[0]),
            "out_y": float(action[1]),
            "out_z": float(action[2]),
            "yaw": float(action[3]),
            "pitch": float(action[4])
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("🚀 드론 자율비행 통신소 서버가 가동되었습니다!")
    print("기다리고 있습니다... 언리얼 엔진 시뮬레이션을 실행해주세요 (포트: 5000)")
    # 내 컴퓨터(로컬)에서 5000번 포트로 서버를 엽니다.
    app.run(host='0.0.0.0', port=5000)