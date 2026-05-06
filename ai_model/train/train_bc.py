import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os

# 신경망 설계 (입력 6개 -> 출력 5개)
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

def start_train():
    # 🌟 RTX 1660 GPU 가동 설정! (이 부분이 추가되었습니다)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 학습 장치: {device} (RTX 1660 가동 준비 완료!)")

    # 1. 데이터 불러오기 (헤더 오류 방지 처리)
    file_path = 'data/raw_data.csv'
    
    # 깨진 첫 줄을 무시하고 데이터를 읽어온 뒤, 이름을 수동으로 지정합니다.
    data = pd.read_csv(file_path, skiprows=[0], header=None)
    data.columns = [
        'timestamp', 'rel_x', 'rel_y', 'rel_z', 'threat', 'hp', 
        'is_damaged', 'out_x', 'out_y', 'out_z', 'yaw', 'pitch'
    ]

    # 2. 입력(X)과 출력(y) 분리
    X = data[['rel_x', 'rel_y', 'rel_z', 'threat', 'hp', 'is_damaged']].values
    y = data[['out_x', 'out_y', 'out_z', 'yaw', 'pitch']].values

    # 3. 텐서 변환 및 로더 설정
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    # 4. 학습 설정 (🌟 모델을 GPU로 전송!)
    model = BCNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. 학습 루프
    print("\n🔥 실데이터 기반 AI 학습을 시작합니다...")
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in loader:
            # 🌟 데이터도 GPU로 전송해서 초고속 계산!
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 10번마다 결과 출력
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"▶ 에포크 {epoch+1}/{epochs} 완료, 평균 손실(Loss): {avg_loss:.4f}")

    # 6. 모델 저장 (폴더가 없으면 생성)
    if not os.path.exists('model'):
        os.makedirs('model')
        
    torch.save(model.state_dict(), 'model/bc_model.pth')
    print("\n🎉 학습 완료! 완성된 뇌가 'model/bc_model.pth' 에 저장되었습니다.")

if __name__ == "__main__":
    start_train()