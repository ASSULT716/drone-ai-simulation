import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import os

try:
    from ai_model.networks import Actor, Critic
except ModuleNotFoundError:
    from networks import Actor, Critic

class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=200000)
        
        # 하이퍼파라미터 세팅
        self.gamma = 0.99           # 미래 보상 할인율
        self.tau = 0.005            # 타겟 네트워크 소프트 업데이트 비율
        self.alpha = 0.2            # 엔트로피(창의성) 가중치
        self.batch_size = 256       # 한 번에 복습할 시험지 갯수
        
        # 🧠 SAC 핵심: 비평가(Critic)의 예측 안정성을 위해 타겟 네트워크 2개 추가
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        
        # 초기 가중치 동기화
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
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
        # 🌟 안전장치: 버퍼에 데이터가 충분히 쌓여 배치 사이즈를 채울 수 있을 때만 학습
        if len(self.memory) < self.batch_size:
            return
        
        # 1. 리플레이 버퍼에서 랜덤 샘플링
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([t[0] for t in mini_batch])
        actions = torch.FloatTensor([t[1] for t in mini_batch])
        rewards = torch.FloatTensor([t[2] for t in mini_batch]).unsqueeze(1)
        next_states = torch.FloatTensor([t[3] for t in mini_batch])
        dones = torch.FloatTensor([t[4] for t in mini_batch]).unsqueeze(1)

        # 2. Critic (비평가) 업데이트 계산
        with torch.no_grad():
            # 다음 상태에서의 행동 예측
            next_actions = self.actor(next_states)
            # 타겟 Critic 2개 중 더 작은 Q값을 선택하여 과대평가 방지 (SAC 핵심 수식)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            # 벨만 방정식 적용한 가치 점수 정답지 생성
            expected_q = rewards + (1 - dones) * self.gamma * target_q

        # 현재 Critic의 예측값 계산 및 오차(Loss) 백프로퍼게이션
        curr_q1 = self.critic1(states, actions)
        curr_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(curr_q1, expected_q)
        critic2_loss = F.mse_loss(curr_q2, expected_q)
        
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        # 3. Actor (행위자) 업데이트 계산
        # 현재 상태에서 Actor가 취할 최적의 행동 점수를 Critic에게 물어봄
        pred_actions = self.actor(states)
        actor_q1 = self.critic1(states, pred_actions)
        actor_q2 = self.critic2(states, pred_actions)
        actor_q = torch.min(actor_q1, actor_q2)
        
        # 보상 점수를 높이는 방향으로 학습 (Gradient Ascent를 위해 마이너스 부호)
        actor_loss = -actor_q.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 4. Target Critic 네트워크 소프트 업데이트 (안정적인 학습 유지)
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
    def save_models(self, path="model/"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}sac_actor.pth")
        torch.save(self.critic1.state_dict(), f"{path}sac_critic.pth")
        print("✅ SAC 모델 뇌 가중치 파일(.pth) 저장 성공!")
        
    def load_models(self, path="model/"):
        try:
            self.actor.load_state_dict(torch.load(f"{path}sac_actor.pth"))
            print("🔄 사전 학습 모델 로드 성공!")
        except:
            print("⚠️ 저장된 모델이 없어 랜덤 가중치로 시작합니다.")