import os
import time
import struct
import numpy as np
import glob
from pythonosc import udp_client

# SAC 에이전트 불러오기
try:
    from sac_agent import SACAgent
except ModuleNotFoundError:
    from ai_model.sac_agent import SACAgent

# ==========================================
# 📡 1. 발신 설정 (OSC -> 언리얼)
# ==========================================
UNREAL_IP = "100.82.247.96"  
OSC_PORT = 7000          
osc_client = udp_client.SimpleUDPClient(UNREAL_IP, OSC_PORT)

# ==========================================
# 📂 2. 수신 설정 (언리얼 BIN 파일 폴더 경로)
# ==========================================
BIN_DIR_PATH = r"C:\Users\user\Desktop\Projects\drone-ai-simulation\data\RL_Data\Tutorial\Packet"

def run_server():
    print(f"🚀 [Online RL] 드론 AI 기지국 가동 시작...")
    print(f"📡 [발신] OSC Action ➔ {UNREAL_IP}:{OSC_PORT} (/drone/input)")
    print(f"📂 [수신] 폴더 실시간 감시 중 ➔ {BIN_DIR_PATH}")
    
    agent = SACAgent(state_dim=59, action_dim=8)
    
    model_path = os.path.join("ai_model", "model")
    if os.path.exists(model_path):
        agent.load_models(path=model_path + "/")
        print("✅ 기존 AI 뇌(가중치) 로드 완료.")
    else:
        print("⚠️ 저장된 모델이 없어 초기 가중치로 시작합니다.")

    # ==========================================
    # 📥 사전 학습 데이터(NPZ) 로드
    # ==========================================
    data_dir = r"C:\Users\user\Desktop\Projects\drone-ai-simulation\data\processed_data"
    
    if os.path.exists(data_dir):
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        if len(npz_files) > 0:
            print(f"📥 총 {len(npz_files)}개의 사전 학습 데이터(NPZ) 로딩 시작...")
            total_loaded = 0
            
            for file_path in npz_files:
                try:
                    data = np.load(file_path)
                    states, actions, rewards, next_states = data['states'], data['actions'], data['rewards'], data['next_states']
                    
                    dones = data.get('dones', data.get('done', np.zeros(len(states))))
                    if not ('dones' in data or 'done' in data):
                        dones[-1] = 1.0 

                    for i in range(len(states)):
                        agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    
                    total_loaded += len(states)
                except Exception as e:
                    print(f"🚨 {os.path.basename(file_path)} 로드 에러: {e}")
            
            print(f"✅ 총 {total_loaded}개의 조종 경험 적재 완료.")
        else:
            print(f"ℹ️ {data_dir} 폴더에 NPZ 파일이 없습니다.")
    else:
        print(f"🚨 지정한 데이터 경로를 찾을 수 없습니다: {data_dir}")

    # ==========================================
    # 🔄 실시간 폴더 감시 및 추론 루프
    # ==========================================
    last_state = None
    last_action = None
    
    last_processed_file = None
    last_mtime = 0
    last_size = 0 
    
    episode_start_time = None  # 시동을 걸기 위한 에피소드 타이머
    backspace_hold_until = 0.0 # 💡 [추가] 백스페이스(리트라이) 꾹 누르기 유지 타이머
    
    print("\n⏳ 언리얼 엔진이 폴더에 bin 파일을 생성하기를 기다리는 중...")
    os.makedirs(BIN_DIR_PATH, exist_ok=True)

# ==========================================
    # 🔄 실시간 폴더 감시 및 추론 루프
    # ==========================================
    last_state = None
    last_action = None
    
    last_processed_file = None
    last_mtime = 0
    last_size = 0 
    
    episode_start_time = None  
    backspace_hold_until = 0.0 
    engine_started = False  # 💡 [추가] 이번 에피소드에서 시동을 걸었는지 확인하는 변수
    
    os.makedirs(BIN_DIR_PATH, exist_ok=True)

    # ==========================================
    # 🧹 [추가] 서버 시작 전 남아있는 과거 파일 찌꺼기 삭제
    # ==========================================
    print("\n🧹 서버 가동 전 폴더 내 기존 bin 파일을 정리합니다...")
    old_files = glob.glob(os.path.join(BIN_DIR_PATH, "*.bin"))
    for old_f in old_files:
        try:
            os.remove(old_f)
        except Exception:
            pass
    print("✨ 정리 완료! 새로운 실시간 패킷을 기다립니다.")
    # ==========================================

    print("\n⏳ 언리얼 엔진이 폴더에 bin 파일을 생성하기를 기다리는 중...")

    while True:
        try:
            bin_files = glob.glob(os.path.join(BIN_DIR_PATH, "*.bin"))
            
            if not bin_files:
                time.sleep(0.01)
                continue
                
            latest_file = max(bin_files, key=os.path.getmtime)
            mtime = os.path.getmtime(latest_file)
            fsize = os.path.getsize(latest_file)
                
            with open(latest_file, 'rb') as f:
                if fsize < 248:
                    time.sleep(0.005)
                    continue
                
                f.seek(fsize - 248)
                data = f.read(248)
                
                if len(data) == 248:
                    state_tuple = struct.unpack('<62f', data)
                    raw_data = np.array(state_tuple, dtype=np.float32)
                    
                    current_step = raw_data[0]
                    done_state = raw_data[1] 
                    step_reward = raw_data[2]
                    current_state = raw_data[3:]
                    
                    # 첫 패킷을 받은 순간부터 시간 측정 시작
                    current_sys_time = time.time()
                    if episode_start_time is None:
                        episode_start_time = current_sys_time
                    
                    elapsed_time = current_sys_time - episode_start_time
                    
                    if last_state is not None:
                        is_done = 1.0 if done_state != 0 else 0.0
                        agent.store_transition(last_state, last_action, step_reward, current_state, is_done)
                        if len(agent.memory) > agent.batch_size:
                            agent.train_step() 
                    
                    if done_state != 0:
                        last_state = None
                        last_action = None
                        last_processed_file = latest_file
                        last_mtime = mtime
                        last_size = fsize
                        episode_start_time = None  # 에피소드 종료/재시작 시 타이머 리셋
                        backspace_hold_until = 0.0 # 💡 [추가] 에피소드가 끝나면 백스페이스 누름 상태도 뗌(초기화)
                        engine_started = False     # 💡 [추가] 다음 에피소드를 위해 시동 변수 리셋
                        continue
                    
                    # ==========================================
                    # 🧠 1. AI Action 추론
                    # ==========================================
                    action = agent.select_action(current_state)
                    
                    # ==========================================
                    # 🛠️ 2. Action 후처리 (언리얼 요구 규격 및 타이머 적용)
                    # ==========================================
                    processed_action = np.zeros(8, dtype=np.float32)

                    # 1~4. Move Up, Right, Forward, Input Roll (-1, 0, 1 중 하나)
                    processed_action[0] = np.round(action[0])
                    processed_action[1] = np.round(action[1])
                    processed_action[2] = np.round(action[2])
                    processed_action[3] = np.round(action[3])

                    # 5~6. Input Look up, Input Turn (-1 ~ 1 연속값 유지)
                    processed_action[4] = action[4]
                    processed_action[5] = action[5]

                    # ==========================================
                    # 💡 7. Pressing_I (AI 제어 차단, 1.6초 후 단 한 번만 1 출력)
                    # ==========================================
                    if elapsed_time >= 1.6 and not engine_started:
                        processed_action[6] = 1.0  # 1.6초가 지났고 시동을 안 걸었다면 1 출력
                        engine_started = True      # 시동을 걸었다고 상태 업데이트
                    else:
                        processed_action[6] = 0.0  # 그 외의 모든 순간(1.6초 이전, 또는 시동 건 직후)에는 0 유지

                    # ==========================================
                    # 💡 8. Is_Pressing_Backspace (리트라이 꾹 누르기 강제 유지)
                    # AI가 단 1프레임이라도 백스페이스를 눌렀다면, 지정된 시간 동안 강제로 유지시킴
                    # ==========================================
                    if action[7] > 0:
                        # 누르겠다고 판단한 순간부터 2.5초 동안 유지 타이머 설정
                        backspace_hold_until = current_sys_time + 2.5
                        
                    if current_sys_time <= backspace_hold_until:
                        processed_action[7] = 1.0
                    else:
                        processed_action[7] = 0.0
                    
                    # 통신용 리스트 변환
                    action_list = processed_action.tolist()

                    # ==========================================
                    # 🔍 3. [이현수님 디버깅 코드]
                    # ==========================================
                    try:
                        import datetime
                        current_time_str = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        formatted_list = [round(float(x), 4) for x in action_list]
                        print(f"[{current_time_str}] [디버깅] 개수: {len(formatted_list)}개 | 데이터: {formatted_list}")
                        print(f"⏱️ [에피소드 경과 시간] {elapsed_time:.2f}초 경과")
                        
                        # 백스페이스가 유지 중인지 확인하기 위한 로그
                        if processed_action[7] == 1.0:
                            print(f"🔄 [알림] 리트라이(Backspace) 꾹 누르는 중! (남은 유지 시간: {max(0, backspace_hold_until - current_sys_time):.2f}초)")
                            
                    except Exception as e:
                        print(f"[디버깅 에러] 출력 실패: {e}")

                    # 상태 및 액션 업데이트 (다음 스텝 버퍼 저장을 위함)
                    last_state = current_state
                    last_action = action  # SAC 훈련 시 역전파를 위해 원본 action 저장
                    
                    # ==========================================
                    print("\n" + "="*40)
                    print(f"🧠 AI 모델 추론 완료! (Step: {int(current_step)})")
                    print("📊 [언리얼로 전송되는 8개의 Action 값]")
                    print(f"  ▶ [0] Move Up       : {action_list[0]:>8.4f} (-1, 0, 1)")
                    print(f"  ▶ [1] Move Right    : {action_list[1]:>8.4f} (-1, 0, 1)")
                    print(f"  ▶ [2] Move Forward  : {action_list[2]:>8.4f} (-1, 0, 1)")
                    print(f"  ▶ [3] Input Roll    : {action_list[3]:>8.4f} (-1, 0, 1)")
                    print(f"  ▶ [4] Input Look Up : {action_list[4]:>8.4f} (-1 ~ 1)")
                    print(f"  ▶ [5] Input Turn    : {action_list[5]:>8.4f} (-1 ~ 1)")
                    print(f"  ▶ [6] Pressing_I    : {action_list[6]:>8.4f} (0, 1)")
                    print(f"  ▶ [7] Backspace     : {action_list[7]:>8.4f} (0, 1)")
                    print("="*40 + "\n")
                    # ==========================================

                    osc_client.send_message("/drone/input", action_list)
                    
                    print(f"🚀 [OSC 연속 송신 중...] 언리얼이 받을 때까지 쏘는 중 (파일: {os.path.basename(latest_file)})")
                    
            last_processed_file = latest_file
            last_mtime = mtime
            last_size = fsize
            
        except Exception as e:
            print(f"🚨 [루프 에러 발생] {e}")
            pass
            
        # 과부하 방지를 위해 루프 속도를 0.05초로 변경
        time.sleep(0.05)

if __name__ == "__main__":
    run_server()