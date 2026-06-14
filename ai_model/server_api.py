import socket
import struct
import os
import numpy as np
from datetime import datetime

from config import (MAX_SPEED, MAX_STEPS, LIDAR_H_MAX, LIDAR_D_MAX, MAX_ENEMY_DIST, 
                    MAX_DEST_DIST, MAX_THREAT_LEVEL, MAX_ENEMY_TYPE, MAX_BOUNDARY_XY, 
                    MAX_CEILING, REWARD_MAX)

# 🌟 62차원 입력, 8차원 출력으로 스펙 업그레이드
IN_PACKET_DIM = 62
IN_PACKET_SIZE = IN_PACKET_DIM * 4
OUT_PACKET_DIM = 8

LOG_DIR = r"D:\Developement\DevProject\drone-ai-simulation\ai_model\AI_Control_Packet"

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def preprocess_live_obs(raw):
    obs = np.array(raw, dtype=np.float32)

    # 0~2: 스텝, 상태, 보상
    obs[0] = np.clip(obs[0], 0, MAX_STEPS) / MAX_STEPS
    obs[1] = np.clip(obs[1], 0, 3) / 3.0
    obs[2] = np.clip(obs[2], -REWARD_MAX, REWARD_MAX) / REWARD_MAX

    # 3~7: 경계선 및 천장
    obs[3:7] = np.clip(obs[3:7], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY
    obs[7] = np.clip(obs[7], 0.0, MAX_CEILING) / MAX_CEILING

    # 8~10: 속도 / 11~13: 자세각
    obs[8:11] = np.clip(obs[8:11], -MAX_SPEED, MAX_SPEED) / MAX_SPEED
    obs[11:14] = obs[11:14] / 180.0

    # 14~17: 하향 라이다 및 목적지
    obs[14] = np.clip(obs[14], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX
    obs[15] = np.clip(obs[15], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST
    obs[16] = (((obs[16] + 180.0) % 360.0) - 180.0) / 180.0
    obs[17] = (((obs[17] + 180.0) % 360.0) - 180.0) / 180.0

    # 18~22: Boolean 플래그 (Is_On, Ready, Press_I, Is_AI, Is_Backspace) -> 스케일링 패스

    # 🌟 23~37: 적 위협 정보 (인덱스 +1 쉬프트 적용)
    for i in range(3):
        idx = 23 + (i * 5)
        obs[idx] = np.clip(obs[idx], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
        obs[idx+1:idx+4] = np.clip(obs[idx+1:idx+4], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
        obs[idx+4] = np.clip(obs[idx+4], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL

    # 🌟 38~61: 수평 24방향 라이다 (인덱스 +1 쉬프트 적용)
    obs[38:62] = np.clip(obs[38:62], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX

    return obs

def run_socket_server():
    host = '0.0.0.0'
    port = 5000
    
    os.makedirs(LOG_DIR, exist_ok=True)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"🚀 [Online RL] 실시간 자율비행 및 학습 기지국 포트 {port} 가동 중...")

    # 🌟 1. AI 에이전트 인스턴스화 (62D State, 8D Action)
    agent = SACAgent(state_dim=62, action_dim=8)
    agent.load_models(path="model/") # 기존 가중치가 있다면 로드

    has_pressed_i = False  
    
    # 🌟 2. 시간차(Transition) 기억을 위한 임시 저장소
    last_state = None
    last_action = None
    
    log_file = None
    current_log_hour = -1

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🔌 언리얼 드론 원격 접속 성공: {addr}")

        try:
            while True:
                data = recvall(client_socket, IN_PACKET_SIZE)
                if not data:
                    print("🚨 통신 두절, 서버 안전 정지 실행")
                    client_socket.sendall(struct.pack(f'<{OUT_PACKET_DIM}f', *[0.0]*8))
                    break

                raw_array = struct.unpack(f'<{IN_PACKET_DIM}f', data)
                
                current_step = int(raw_array[0])
                done_state = int(raw_array[1]) 
                current_reward = float(raw_array[2]) # 🌟 방금 전 행동으로 얻은 보상

                scaled_obs = preprocess_live_obs(raw_array)

                # ========================================================
                # 💀 사망 / 에피소드 종료 처리 로직
                # ========================================================
                if done_state > 0:
                    print(f"🚨 [종료/사망] 스텝: {current_step} | 상태: {done_state} | 보상: {current_reward}")
                    
                    # 🌟 과거의 행동이 '사망'이라는 결과를 낳았으므로 버퍼에 페널티 기록
                    if last_state is not None and last_action is not None:
                        agent.store_transition(last_state, last_action, current_reward, scaled_obs, done=True)
                        agent.train_step() # 사망 시 모델 업데이트
                    
                    # 드론이 리스폰되므로, 과거의 기억을 끊어줌 (다음 스텝과 연결 방지)
                    last_state = None
                    last_action = None
                    has_pressed_i = False
                    
                    final_action = [0.0] * 8
                    response_data = struct.pack(f'<{OUT_PACKET_DIM}f', *final_action)
                    client_socket.sendall(response_data)
                    continue 

                # ========================================================
                # ✈️ 정상 비행 및 실시간 학습 로직
                # ========================================================
                is_engine_on = (raw_array[18] >= 0.5)
                is_ai_controlled = (raw_array[21] >= 0.5)
                
                if current_step < 32:
                    has_pressed_i = False

                if not is_ai_controlled:
                    # 사람이 조종할 때는 학습 데이터가 오염되지 않도록 기억 초기화
                    last_state = None
                    last_action = None
                    
                    final_action = [0.0] * 8
                    if is_engine_on: has_pressed_i = True
                    else: has_pressed_i = False 
                else:
                    # 🌟 [학습 파트] 이전 스텝의 데이터를 버퍼에 저장
                    if last_state is not None and last_action is not None:
                        agent.store_transition(last_state, last_action, current_reward, scaled_obs, done=False)
                        agent.train_step() # 실시간 역전파(Backprop) 진행
                    
                    # 🌟 [추론 파트] 새로운 상태를 보고 행동 예측
                    # 주의: 실시간 학습 시에는 select_action() 내부에 노이즈(Exploration)가 섞여 있어야 합니다.
                    action_from_ai = agent.select_action(scaled_obs) 
                    
                    pressing_i = 0.0
                    if not has_pressed_i and current_step >= 32:
                        pressing_i = 1.0
                        has_pressed_i = True
                    
                    action_from_ai[6] = pressing_i # 시동 강제 오버라이드
                    final_action = action_from_ai
                    
                    # 🌟 다음 루프를 위해 현재 상태와 행동을 기억해둠
                    last_state = scaled_obs.copy()
                    last_action = final_action.copy()

                # 패킷 전송
                response_data = struct.pack(f'<{OUT_PACKET_DIM}f', *final_action)
                client_socket.sendall(response_data)
                
                if is_ai_controlled:
                    now = datetime.now()
                    if now.hour != current_log_hour:
                        if log_file: log_file.close()
                        current_log_hour = now.hour
                        filename = f"ai_action_{now.strftime('%Y%m%d_%H%M%S')}.bin"
                        filepath = os.path.join(LOG_DIR, filename)
                        log_file = open(filepath, 'wb')
                    
                    if log_file:
                        log_file.write(response_data)

        except ConnectionResetError:
            print("🚨 언리얼 시뮬레이터 연결이 끊어졌습니다.")
        finally:
            client_socket.close()
            if log_file: log_file.close()

if __name__ == '__main__':
    run_socket_server()