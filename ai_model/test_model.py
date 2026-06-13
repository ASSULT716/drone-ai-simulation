import torch
import torch.nn as nn
import numpy as np

# 1. 모델 구조 정의 (학습 때와 동일해야 함)
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

def test_inference():
    # 2. 모델 로드
    model = BCNet()
    model.load_state_dict(torch.load('model/bc_model.pth'))
    model.eval() # 평가 모드 전환

    # 3. 가상의 테스트 데이터 (적의 상대 위치, 위협 수준, HP, 데미지 여부)
    # 예: 적이 내 앞(z=10.0)에 있고, 위협적(0.9)이며, 내 피는 80인 상황
    test_input = torch.tensor([[0.0, 0.0, 10.0, 0.9, 80.0, 0]], dtype=torch.float32)

    # 4. 예측 실행
    with torch.no_grad():
        prediction = model(test_input)
    
    out = prediction[0].numpy()
    
    print("=== AI 모델 추론 테스트 ===")
    print(f"입력 상황: 적이 정면 10m 위치, 위협도 0.9, HP 80")
    print("-" * 30)
    print(f"AI의 판단 (예측값):")
    print(f"▶ 이동 방향 (x, y, z): {out[0]:.2f}, {out[1]:.2f}, {out[2]:.2f}")
    print(f"▶ 시선 회전 (Yaw, Pitch): {out[3]:.2f}, {out[4]:.2f}")
    print("==========================")

if __name__ == "__main__":
    test_inference()