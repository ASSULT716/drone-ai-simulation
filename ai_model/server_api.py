import socket
import struct
import numpy as np
# from sac_agent import SACAgent
# from sac_model import DroneEngineAutoStartWrapper

# 🌟 수정 완료: State 56개 * 4바이트 = 총 224바이트 수신 대기
STATE_DIM = 56
PACKET_SIZE = STATE_DIM * 4

def run_socket_server():
    host = '0.0.0.0'
    port = 5000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"🚀 [초고속 소켓 서버 가동] 포트 {port}에서 224바이트(56차원) 패킷 대기 중...")
    
    # agent = SACAgent()
    # agent.load_models("models/sac_drone_base.zip")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"🔌 언리얼 엔진 접속 완료: {addr}")

        try:
            while True:
                # 1. 224바이트 패킷 수신
                data = client_socket.recv(PACKET_SIZE)
                if not data:
                    break

                if len(data) == PACKET_SIZE:
                    # 2. 224바이트 ➔ 56개의 실수(Float)로 즉시 언팩
                    raw_state_array = struct.unpack(f'{STATE_DIM}f', data)
                    
                    # 3. (래퍼 전처리 통과)
                    # processed_state = env._preprocess_obs(raw_state_array)
                    
                    # 4. AI 액션 도출 (출력 6개!)
                    # action_from_ai = agent.select_action(processed_state)
                    action_from_ai = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 임시 6차원 스틱 입력
                    
                    # 5. 시동 제어 플래그(Cmd_Engine) 추가 ➔ 총 7개의 숫자 전송
                    # (만약 언리얼에서 엔진 시동 명령을 안 받는 구조라면 이 부분을 수정하면 됩니다)
                    final_action = action_from_ai + [1.0] 
                    
                    # 6. 액션(7개*4바이트 = 28바이트)을 팩(Pack)해서 언리얼로 리턴!
                    response_data = struct.pack('7f', *final_action)
                    client_socket.sendall(response_data)
                else:
                    print(f"⚠️ 패킷 누락 발생: {len(data)} 바이트 수신됨")
                    
        except ConnectionResetError:
            print("🚨 언리얼 엔진 시뮬레이터가 종료되었거나 연결이 끊어졌습니다.")
        finally:
            client_socket.close()

if __name__ == '__main__':
    run_socket_server()