import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# 1. 커스텀 데이터셋 구현 (Data Loader의 심장)
# ---------------------------------------------------------
class DroneDefenseDataset(Dataset):
    def __init__(self, num_samples=1000):
        # 입력(X): 6개 피처 (rel_x, rel_y, rel_z, threat, hp, is_damaged)
        self.x_data = torch.rand(num_samples, 6, dtype=torch.float32)
        # 출력(Y): 5개 피처 (out_x, out_y, out_z, yaw, pitch)
        self.y_data = torch.rand(num_samples, 5, dtype=torch.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# ---------------------------------------------------------
# 2. 모델 정의 (기존에 만든 BCNet)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. 학습 테스트 루프
# ---------------------------------------------------------
def test_training_pipeline():
    print("🚀 3주차: 데이터 로더 및 더미 데이터 학습 테스트 시작\n")

    # 데이터 준비
    dataset = DroneDefenseDataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"✅ 데이터 로더 세팅 완료: 총 {len(dataset)}개의 더미 데이터 준비됨.")

    # 모델 및 학습 도구 설정
    model = BCNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 미니 학습 루프 (10번 반복)
    epochs = 10
    print("✅ 모델 학습 루프 가동 중...")
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()       
            pred = model(batch_x)       
            loss = criterion(pred, batch_y) 
            loss.backward()             
            optimizer.step()            
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"▶ 에포크 [{epoch+1}/{epochs}] - 평균 손실(Loss): {avg_loss:.4f}")

    print("\n🎉 파이프라인 테스트 완료! 모델이 더미 데이터를 정상적으로 학습합니다.")

if __name__ == "__main__":
    test_training_pipeline()