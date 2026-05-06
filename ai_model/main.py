import os
from utils.logger import save_log

# 1. 가짜 데이터 10줄 생성 (테스트용)
DATA_PATH = 'data/raw_data.csv'
print("데이터 수집 테스트 시작...")

for i in range(10):
    # 가짜 상태: [x, y, z, threat, hp, is_damaged]
    dummy_state = [0.1*i, 0.2*i, 0.5, 0.8, 100-i, 0]
    # 가짜 행동: [mx, my, mz, yaw, pitch]
    dummy_action = [1, 0, 0, 0.01, 0.02]
    
    save_log(DATA_PATH, dummy_state, dummy_action)

print(f"{DATA_PATH}에 데이터가 저장되었습니다.")
print("이제 터미널에서 'python train/train_bc.py'를 입력해 학습을 시도해보세요.")