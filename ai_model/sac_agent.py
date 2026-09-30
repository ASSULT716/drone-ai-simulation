import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from networks import GaussianPolicy, Critic

class SACAgent:
    def __init__(self, state_dim=61, action_dim=8, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 네트워크 초기화
        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # 타겟 네트워크를 현재 네트워크와 동일하게 동기화
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 2. Adam 옵티마이저 설정 (학습률 3e-4)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        
        # 3. 자동 엔트로피(Alpha) 튜닝 설정
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        
        # 4. Epsilon-Greedy 설정 (초기 학습 시 모드 붕괴 방지용 강제 탐험)
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def select_action(self, state, evaluate=False):
        """환경과 상호작용하기 위한 행동 선택 로직"""
        # 1. Epsilon-Greedy 무작위 탐험 (추락 패턴 파괴)
        if not evaluate and np.random.rand() < self.epsilon:
            # -1.0 ~ 1.0 사이의 완전 무작위 8차원 행동 반환
            return np.random.uniform(-1.0, 1.0, size=(8,))
            
        # 2. 신경망에 의한 행동 선택
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state) # 평가 시에는 평균값(확정) 사용
            else:
                action, _, _ = self.policy.sample(state) # 학습 시에는 분포에서 샘플링
                
        return action.cpu().data.numpy().flatten()
        
    def update_epsilon(self):
        """에피소드 종료 시 epsilon 값 감소"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_parameters(self, memory, batch_size):
        """리플레이 버퍼에서 미니배치를 추출하여 신경망 학습 (BPTT 대신 일반 Backprop 사용)"""
        # 메모리 버퍼에서 샘플링 (구현된 ReplayBuffer 클래스 사용 가정)
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)

        # [Critic 업데이트 로직 - 벨만 방정식 적용]
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.log_alpha.exp() * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # [Actor 업데이트 로직]
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.log_alpha.exp() * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # [Alpha(엔트로피) 업데이트 로직]
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # [Target Critic 소프트 업데이트]
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)