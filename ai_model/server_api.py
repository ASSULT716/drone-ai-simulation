import socket
import struct
import os
import numpy as np
from datetime import datetime

# 🌟 데이터 파트 동기화 상수
from config import (MAX_SPEED, MAX_STEPS, LIDAR_H_MAX, LIDAR_D_MAX, MAX_ENEMY_DIST, 
                    MAX_DEST_DIST, MAX_THREAT_LEVEL, MAX_ENEMY_TYPE, MAX_BOUNDARY_XY, 
                    MAX_CEILING, REWARD_MAX)

IN_PACKET_DIM = 61
IN_PACKET_SIZE = IN_PACKET_DIM * 4
OUT_PACKET_DIM = 7

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
    obs[0] = np.clip(obs[0], 0, MAX_STEPS) / MAX_STEPS
    obs[1] = np.clip(obs[1], 0, 3) / 3.0
    obs[2] = np.clip(obs[2], -REWARD_MAX, REWARD_MAX) / REWARD_MAX
    obs[3:7] = np.clip(obs[3:7], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY
    obs[7] = np.clip(obs[7], 0.0, MAX_CEILING) / MAX_CEILING
    obs[8:11] = np.clip(obs[8:11], -MAX_SPEED, MAX_SPEED) / MAX_SPEED
    obs[11:14] = obs[11:14] / 180.0
    obs[14] = np.clip(obs[14], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX
    obs[15] = np.clip(obs[15], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST
    obs[16] = (((obs[16] + 180.0) % 360.0) - 180.0) / 180.0
    obs[17] = (((obs[17] + 180.0) % 360.0) - 180.0) / 180.0
    for i in range(3):
        idx = 22 + (i * 5)
        obs[idx] = np.clip(obs[idx], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
        obs[idx+1:idx+4] = np.clip(obs[idx+1:idx+4], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
        obs[idx+4] = np.clip(obs[idx+4], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL
    obs[37:61] = np.clip(obs[37:61], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX
    return obs

def run_socket_server():
    host = '0.0.0.0'
    port = 5000
    
    os.makedirs(LOG_DIR, exist_ok=True)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"🚀 [61D 동기화 완료] 실시간 자율비행 기지국 포트 {port} 가동 중...")

    has_pressed_i = False  
    log_file = None
    current_log_hour = -1

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🔌 언리얼 드론 원격 접속 성공: {addr}")

        try:
            while True:
                data = recvall(client_socket, IN_PACKET_SIZE)
                if not data:
                    break

                # 🌟 바이트 순서 명시 (<: Little-Endian)
                raw_array = struct.unpack(f'<{IN_PACKET_DIM}f', data)
                
                current_step = int(raw_array[0])
                done_state = int(raw_array[1]) 

                # 🌟 리트라이 및 사망 로직 (Early Exit)
                if done_state > 0:
                    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"🚨 [사망 발생] 시간: {now_str} | 스텝: {current_step} | 상태: {done_state}")
                    
                    has_pressed_i = False
                    
                    final_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # 🌟 여기도 바이트 순서 명시 추가됨
                    response_data = struct.pack(f'<{OUT_PACKET_DIM}f', *final_action)
                    client_socket.sendall(response_data)
                    continue 

                is_engine_on = (raw_array[18] >= 0.5)
                is_ai_controlled = (raw_array[21] >= 0.5)
                
                if current_step < 32:
                    has_pressed_i = False

                if not is_ai_controlled:
                    final_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if is_engine_on:
                        has_pressed_i = True
                    else:
                        has_pressed_i = False 
                else:
                    scaled_obs = preprocess_live_obs(raw_array)
                    
                    # [TODO: 실제 모델 연동]
                    action_from_ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
                    
                    pressing_i = 0.0
                    if not has_pressed_i and current_step >= 32:
                        pressing_i = 1.0
                        has_pressed_i = True
                    
                    final_action = list(action_from_ai) + [pressing_i]

                # 🌟 최종 패킷 전송에도 바이트 순서 통일
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