import numpy as np
import torch
import torch.optim as optim
from collections import deque
import os

# ai_model 폴더 내에 있다고 가정하고 import 경로 설정
try:
    from ai_model.networks import Actor, Critic
except ModuleNotFoundError:
    from networks import Actor, Critic

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=200000) # 대규모 오프라인 데이터 수용
        
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=3e-4)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t)
        return action.numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        # 향후 오프라인/온라인 학습 수식이 들어갈 자리
        if len(self.memory) < 1000:
            return
        pass

    def save_models(self, path="model/"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}sac_actor.pth")
        torch.save(self.critic1.state_dict(), f"{path}sac_critic.pth")
        print("✅ 오프라인 모델 가중치 저장 완료!")
        
    def load_models(self, path="model/"):
        try:
            self.actor.load_state_dict(torch.load(f"{path}sac_actor.pth"))
            print("🔄 오프라인 사전 학습 모델 로드 성공! (온라인 파인튜닝 준비완료)")
        except:
            print("⚠️ 저장된 모델이 없어 랜덤 가중치로 시작합니다.")