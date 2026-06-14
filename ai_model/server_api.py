import os
import socket
import struct  # 🌟 바이너리 패킹/언패킹을 위해 필수 추가
import numpy as np
from pythonosc import udp_client

# 🌟 SAC 에이전트 불러오기 (실행 경로 문제 방지)
try:
    from sac_agent import SACAgent
except ModuleNotFoundError:
    from ai_model.sac_agent import SACAgent

# ==========================================
# 📡 통신 및 환경 설정
# ==========================================
# 1. 언리얼 엔진으로 액션을 보낼 OSC(UDP) 클라이언트 설정
UNREAL_IP = "127.0.0.1"  # Tailscale 연동 시 언리얼 PC의 100.x.x.x IP 입력
OSC_PORT = 7000          # 언리얼 OSC 수신 포트
osc_client = udp_client.SimpleUDPClient(UNREAL_IP, OSC_PORT)

# 2. 언리얼 엔진으로부터 상태(State)를 받을 TCP 서버 설정
TCP_HOST = '0.0.0.0'     # 모든 접속 허용
TCP_PORT = 5000

def run_server():
    print(f"🚀 [Online RL] 드론 AI 기지국 가동 시작...")
    print(f"📡 [발신] OSC Action ➔ {UNREAL_IP}:{OSC_PORT} (/drone/input)")
    print(f"🔌 [수신] TCP State  ➔ Port {TCP_PORT} 대기 중...")
    
    # AI 에이전트 인스턴스화 (62D State, 8D Action)
    agent = SACAgent(state_dim=62, action_dim=8)
    
    # 저장된 최적의 가중치가 있다면 로드
    model_path = os.path.join("ai_model", "model")
    if os.path.exists(model_path):
        agent.load_models(path=model_path + "/")
        print("✅ 기존 AI 뇌(가중치) 로드 완료.")
    else:
        print("⚠️ 저장된 모델이 없어 초기 가중치로 시작합니다.")

    # TCP 소켓 서버 오픈
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 포트 재사용 설정
    server_socket.bind((TCP_HOST, TCP_PORT))
    server_socket.listen(1)
    
    while True:
        client_socket, addr = server_socket.accept()
        print(f"✅ 언리얼 드론 원격 접속 성공: {addr}")
        
        try:
            while True:
                # 1. 언리얼에서 오는 62D State 데이터 수신 (62 * 4바이트 = 248바이트)
                data = client_socket.recv(248)
                if not data or len(data) < 248:
                    break
                
                # 🌟 [핵심 수정] 바이너리 데이터 주석을 풀고 실제 62개 float 데이터로 변환
                state = struct.unpack('<62f', data)
                state_np = np.array(state, dtype=np.float32)
                
                # 2. 실시간 데이터를 기반으로 AI 모델의 최적 액션 추론
                action = agent.select_action(state_np)
                
                # 데이터 타입을 일반 파이썬 리스트로 변환
                if isinstance(action, np.ndarray):
                    action_list = action.tolist()
                else:
                    action_list = action
                
                """
                🌟 [최종 확정 액션 인덱스 명세서] 🌟
                action_list[0] = move up
                action_list[1] = move right
                action_list[2] = move forward
                action_list[3] = input roll
                action_list[4] = input pitch (peach)
                action_list[5] = input yaw
                action_list[6] = pressing_i
                action_list[7] = is pressing backspace
                """
                
                # 3. 언리얼 블루프린트로 8차원 액션 통째로 전송 (OSC)
                osc_client.send_message("/drone/input", action_list)
                
        except ConnectionResetError:
            print("🚨 통신 끊김. 언리얼 엔진 재연결 대기 중...")
        finally:
            client_socket.close()

if __name__ == "__main__":
    run_server()