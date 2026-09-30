import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 모드 붕괴를 막기 위한 log_std 강제 클리핑 범위
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def weights_init_(m):
    """
    자료의 가중치 초기화 이론을 반영:
    Linear 레이어에 대해 Kaiming He 초기화 (ReLU 계열에 최적화) 적용
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    """Twin Q-Network: 가치(Value)의 과대평가 방지"""
    def __init__(self, state_dim=61, action_dim=8, hidden_dim=256):
        super(Critic, self).__init__()
        
        # Q1 아키텍처
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),  # 자료를 참고하여 죽은 뉴런 방지
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 아키텍처
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class GaussianPolicy(nn.Module):
    """Actor Network: 상태를 받아 행동의 평균과 분산(log_std) 출력"""
    def __init__(self, state_dim=61, action_dim=8, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weights_init_)

    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # 핵심 안전장치: 분산이 0이 되거나 무한대로 발산하는 것을 방지
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.rsample()  # Reparameterization trick (역전파를 위한 샘플링)
        y_t = torch.tanh(x_t)   # 행동을 [-1, 1] 범위로 압축
        action = y_t
        
        # SAC 수식에 따른 log probability 계산 및 보정
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)