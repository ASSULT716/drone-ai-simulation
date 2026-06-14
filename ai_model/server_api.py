# ai_model/server_api.py
import socket
import struct
import os
import numpy as np
from datetime import datetime

# 🌟 데이터 파트와 완벽히 동기화된 단일 진실 공급원 상수 호출
from config import (MAX_SPEED, MAX_STEPS, LIDAR_H_MAX, LIDAR_D_MAX, MAX_ENEMY_DIST, 
                    MAX_DEST_DIST, MAX_THREAT_LEVEL, MAX_ENEMY_TYPE, MAX_BOUNDARY_XY, 
                    MAX_CEILING, REWARD_MAX)

# 🌟 데이터 담당자의 .bin 규격에 맞춰 수신 차원을 61차원(244바이트)으로 철저히 동기화
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
    """
    🌟 호환성 패치: 언리얼 소켓에서 들어온 61차원 Raw 배열을 
    데이터 담당자의 STATE_FEATURES 순서와 1:1 매핑하여 실시간으로 스케일링합니다.
    """
    obs = np.array(raw, dtype=np.float32)

    # Index 0~2: 기본 정보 스케일링
    obs[0] = np.clip(obs[0], 0, MAX_STEPS) / MAX_STEPS
    obs[1] = np.clip(obs[1], 0, 3) / 3.0
    obs[2] = np.clip(obs[2], -REWARD_MAX, REWARD_MAX) / REWARD_MAX

    # Index 3~7: 맵 경계선 및 천장 거리
    obs[3:7] = np.clip(obs[3:7], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY
    obs[7] = np.clip(obs[7], 0.0, MAX_CEILING) / MAX_CEILING

    # Index 8~10: 드론 속도 벡터
    obs[8:11] = np.clip(obs[8:11], -MAX_SPEED, MAX_SPEED) / MAX_SPEED

    # Index 11~13: 자세각 (Roll, Pitch, Yaw)
    obs[11:14] = obs[11:14] / 180.0

    # Index 14~17: 하향 라이다 및 목적지 상대 정보
    obs[14] = np.clip(obs[14], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX
    obs[15] = np.clip(obs[15], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST
    obs[16] = (((obs[16] + 180.0) % 360.0) - 180.0) / 180.0
    obs[17] = (((obs[17] + 180.0) % 360.0) - 180.0) / 180.0

    # Index 18~21: 불리언 및 제어 토글 상태 상태유지 (스케일링 제외)
    # [Is_On_Bool, Is_Ready_To_Fly_Bool, Pressing_I, Is_AI_Controlled]

    # Index 22~36: 적 위협 정보 3대 정밀 스케일링 (5차원 * 3)
    for i in range(3):
        idx = 22 + (i * 5)
        obs[idx] = np.clip(obs[idx], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
        obs[idx+1:idx+4] = np.clip(obs[idx+1:idx+4], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
        obs[idx+4] = np.clip(obs[idx+4], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL

    # Index 37~60: 수평 24방향 라이다 센서
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

                raw_array = struct.unpack(f'{IN_PACKET_DIM}f', data)
                
                current_step = int(raw_array[0])
                is_engine_on = (raw_array[18] >= 0.5)     # 19번째: Is_On_Bool
                is_ai_controlled = (raw_array[21] >= 0.5) # 22번째: Is_AI_Controlled
                
                if current_step < 32:
                    has_pressed_i = False

                if not is_ai_controlled:
                    # 사람이 조종 토글을 켰을 때: AI 개입을 중단하고 즉시 제어권 반납
                    final_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if is_engine_on:
                        has_pressed_i = True
                    else:
                        has_pressed_i = False 
                else:
                    # AI 자율주행 모드: 실시간 감각 변수 스케일링 후 모델 추론
                    scaled_obs = preprocess_live_obs(raw_array)
                    
                    # [실제 모델 가중치 파일 연동 시 적용 단]
                    # action_from_ai, _ = model.predict(scaled_obs, deterministic=True)
                    action_from_ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
                    
                    pressing_i = 0.0
                    if not has_pressed_i and current_step >= 32:
                        pressing_i = 1.0
                        has_pressed_i = True
                    
                    final_action = list(action_from_ai) + [pressing_i]

                response_data = struct.pack(f'{OUT_PACKET_DIM}f', *final_action)
                client_socket.sendall(response_data)
                
                # AI 제어 상태일 때만 고유 디버깅 로그 백업
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