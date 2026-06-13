import socket
import struct
import os
from datetime import datetime

# 수신(60) / 송신(7) 규격 확실하게 고정
IN_PACKET_DIM = 60
IN_PACKET_SIZE = IN_PACKET_DIM * 4
OUT_PACKET_DIM = 7

# AI 액션 로그 저장 절대 경로
LOG_DIR = r"D:\Developement\DevProject\drone-ai-simulation\ai_model\AI_Control_Packet"

def run_socket_server():
    host = '0.0.0.0'
    port = 5000
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"🚀 [20Hz 소켓 & 파일 백업 서버 가동] 포트 {port} 대기 중...")
    print(f"📁 안전 로그 경로: {LOG_DIR}")

    has_pressed_i = False  
    log_file = None
    current_log_hour = -1

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🔌 언리얼 엔진 드론 입장 완료: {addr}")

        try:
            while True:
                data = client_socket.recv(IN_PACKET_SIZE)
                if not data or len(data) != IN_PACKET_SIZE:
                    break

                raw_array = struct.unpack(f'{IN_PACKET_DIM}f', data)
                
                current_step = int(raw_array[0])
                is_manual_mode = bool(raw_array[3])   
                is_engine_on = bool(raw_array[19])     

                # 🌟 0.05초 주기에 맞춰 1.6초 기준점인 '32스텝'으로 완벽 롤백
                if current_step < 32:
                    has_pressed_i = False

                if is_manual_mode:
                    final_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    if is_engine_on:
                        has_pressed_i = True
                else:
                    action_from_ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 실제 AI 모델 추론 조종값 들어갈 자리
                    
                    pressing_i = 0.0
                    # 🌟 32스텝 정각에 정확히 시동 1회 발사 플래그 세팅
                    if not has_pressed_i and current_step >= 32:
                        pressing_i = 1.0
                        has_pressed_i = True
                    
                    final_action = action_from_ai + [pressing_i]

                # 언리얼 엔진으로 즉시 조종 패킷 발사 (28바이트)
                response_data = struct.pack(f'{OUT_PACKET_DIM}f', *final_action)
                client_socket.sendall(response_data)
                
                # ---------------------------------------------------------
                # 시간별 고유 바이너리 데이터 파일 백업 장치
                # ---------------------------------------------------------
                now = datetime.now()
                if now.hour != current_log_hour:
                    if log_file:
                        log_file.close()
                    current_log_hour = now.hour
                    
                    # 🌟 개선: 분/초까지 파일명에 박아서 파일 꼬임 및 오염을 원천 차단합니다.
                    filename = f"ai_action_{now.strftime('%Y%m%d_%H%M%S')}.bin"
                    filepath = os.path.join(LOG_DIR, filename)
                    log_file = open(filepath, 'wb') # 새로운 고유 파일 쓰기 모드
                
                if log_file:
                    log_file.write(response_data)

        except ConnectionResetError:
            print("🚨 언리얼 엔진 시뮬레이터가 강제 종료되었습니다.")
        finally:
            client_socket.close()
            if log_file:
                log_file.close()
                log_file = None
                current_log_hour = -1

if __name__ == '__main__':
    run_socket_server()