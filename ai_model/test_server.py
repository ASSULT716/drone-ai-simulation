import socket
import struct

# 언리얼로 보낼 8개의 테스트 액션 값 (순서대로 0.1, 0.2, 0.3... 0.8)
test_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(1)
print("📡 블루프린트 통신 테스트 서버 가동 중 (포트 5000)...")

while True:
    client_socket, addr = server_socket.accept()
    print(f"🔌 언리얼 엔진 연결됨: {addr}")
    try:
        while True:
            # 언리얼에서 오는 62D 데이터 수신 (62 * 4 = 248바이트)
            data = client_socket.recv(248) 
            if not data: break
            
            # AI 추론 과정 생략하고 바로 고정된 테스트 패킷 32바이트 송신
            response_data = struct.pack('<8f', *test_action)
            client_socket.sendall(response_data)
    except ConnectionResetError:
        print("🚨 연결 끊김. 재대기 중...")