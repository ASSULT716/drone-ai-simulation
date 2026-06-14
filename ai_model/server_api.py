import socket
import struct
import os
import numpy as np
from datetime import datetime
from config import (MAX_SPEED, LIDAR_H_MAX, LIDAR_D_MAX, MAX_ENEMY_DIST, 
                    MAX_CEILING, MAX_BOUNDARY_XY, MAX_ENEMY_TYPE, MAX_THREAT_LEVEL, MAX_DEST_DIST)

# 🌟 수정: 새로운 BIN 규격 61차원 (244 bytes) 완벽 매핑
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
    61차원 통신 패킷을 분해하여, AI가 학습한 '56차원 State Features' 순서대로 완벽히 재배치 및 스케일링
    """
    obs = np.zeros(56, dtype=np.float32)

    # 1. 물리 (0~5): Vel(8,9,10) / RPY(11,12,13)
    obs[0:3] = np.clip(raw[8:11], -MAX_SPEED, MAX_SPEED) / MAX_SPEED
    obs[3:6] = np.array(raw[11:14]) / 180.0

    # 2. 적 좌표 (6~14): E1(23,24,25) / E2(28,29,30) / E3(33,34,35)
    obs[6:9]   = np.clip(raw[23:26], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
    obs[9:12]  = np.clip(raw[28:31], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST
    obs[12:15] = np.clip(raw[33:36], -MAX_ENEMY_DIST, MAX_ENEMY_DIST) / MAX_ENEMY_DIST

    # 3. 적 메타 (15~20): E1_Type(22), Threat(26) / E2_Type(27), Threat(31) / E3_Type(32), Threat(36)
    obs[15] = np.clip(raw[22], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
    obs[16] = np.clip(raw[26], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL
    obs[17] = np.clip(raw[27], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
    obs[18] = np.clip(raw[31], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL
    obs[19] = np.clip(raw[32], 0.0, MAX_ENEMY_TYPE) / MAX_ENEMY_TYPE
    obs[20] = np.clip(raw[36], 0.0, MAX_THREAT_LEVEL) / MAX_THREAT_LEVEL

    # 4. 라이다 수평 24방향 (21~44): (37~60)
    obs[21:45] = np.clip(raw[37:61], 0.0, LIDAR_H_MAX) / LIDAR_H_MAX

    # 5. 라이다 하향 (45): (14)
    obs[45] = np.clip(raw[14], 0.0, LIDAR_D_MAX) / LIDAR_D_MAX

    # 6. 목적지 (46~48): Dist(15) / Yaw(16) / Pitch(17)
    obs[46] = np.clip(raw[15], 0.0, MAX_DEST_DIST) / MAX_DEST_DIST
    obs[47] = (((raw[16] + 180.0) % 360.0) - 180.0) / 180.0
    obs[48] = (((raw[17] + 180.0) % 360.0) - 180.0) / 180.0

    # 7. 상태 불리언 (49~50): Is_on(18) / Ready(19)
    obs[49] = raw[18]
    obs[50] = raw[19]

    # 8. 맵 경계 (51~55): MaxX(3), MinX(4), MaxY(5), MinY(6), Ceiling(7)
    obs[51:55] = np.clip(raw[3:7], -MAX_BOUNDARY_XY, MAX_BOUNDARY_XY) / MAX_BOUNDARY_XY
    obs[55] = np.clip(raw[7], 0.0, MAX_CEILING) / MAX_CEILING

    return obs

def run_socket_server():
    host = '0.0.0.0'
    port = 5000
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"🚀 [20Hz AI 컨트롤 기지국] 포트 {port} 대기 중... (61차원 패킷 대응 완료)")

    has_pressed_i = False  
    log_file = None
    current_log_hour = -1

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🔌 언리얼 드론 접속 완료: {addr}")

        try:
            while True:
                data = recvall(client_socket, IN_PACKET_SIZE)
                if not data:
                    break

                raw_array = struct.unpack(f'{IN_PACKET_DIM}f', data)
                
                current_step = int(raw_array[0])
                is_engine_on = (raw_array[18] >= 0.5)     # BIN 19번째: Is_on_bool
                is_ai_controlled = (raw_array[21] >= 0.5) # BIN 22번째: Is_AI_Controlled
                
                if current_step < 32:
                    has_pressed_i = False

                if not is_ai_controlled:
                    # [사람 수동 제어 모드]
                    final_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    
                    if is_engine_on:
                        has_pressed_i = True
                    else:
                        has_pressed_i = False 
                else:
                    # [AI 자율 제어 모드]
                    scaled_obs = preprocess_live_obs(raw_array)
                    
                    # (추후 AI 모델 연결 시 주석 해제 및 적용)
                    # action_from_ai, _ = model.predict(scaled_obs, deterministic=True)
                    action_from_ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
                    
                    pressing_i = 0.0
                    if not has_pressed_i and current_step >= 32:
                        pressing_i = 1.0
                        has_pressed_i = True
                    
                    final_action = list(action_from_ai) + [pressing_i]

                # 언리얼로 28바이트 응답 전송
                response_data = struct.pack(f'{OUT_PACKET_DIM}f', *final_action)
                client_socket.sendall(response_data)
                
                # 🌟 수정: AI가 조종할 때만 .bin 로그를 기록하여 사람의 액션 기록 원천 차단
                if is_ai_controlled:
                    now = datetime.now()
                    if now.hour != current_log_hour:
                        if log_file:
                            log_file.close()
                        current_log_hour = now.hour
                        filename = f"ai_action_{now.strftime('%Y%m%d_%H%M%S')}.bin"
                        filepath = os.path.join(LOG_DIR, filename)
                        log_file = open(filepath, 'wb')
                    
                    if log_file:
                        log_file.write(response_data)

        except ConnectionResetError:
            print("🚨 통신 종료.")
        finally:
            client_socket.close()
            if log_file:
                log_file.close()
                log_file = None
                current_log_hour = -1

if __name__ == '__main__':
    run_socket_server()