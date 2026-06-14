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
        # 🌟 이제 62(state), 8(action)이 정상적으로 주입됩니다.
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=200000)
        
        self.gamma = 0.99           
        self.tau = 0.005            
        self.alpha = 0.2            
        self.batch_size = 256       
        
        self.actor = Actor(state_dim, action_dim)
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.target_critic1 = Critic(state_dim, action_dim)
        self.target_critic2 = Critic(state_dim, action_dim)
        
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
        if len(self.memory) < self.batch_size:
            return
        
        mini_batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in mini_batch]))
        actions = torch.FloatTensor(np.array([t[1] for t in mini_batch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in mini_batch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t[3] for t in mini_batch]))
        dones = torch.FloatTensor(np.array([t[4] for t in mini_batch])).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            expected_q = rewards + (1 - dones) * self.gamma * target_q

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

        pred_actions = self.actor(states)
        actor_q1 = self.critic1(states, pred_actions)
        actor_q2 = self.critic2(states, pred_actions)
        actor_q = torch.min(actor_q1, actor_q2)
        
        actor_loss = -actor_q.mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

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