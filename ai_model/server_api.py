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
# 💡 단일 파일이 아닌 '폴더' 전체를 지정합니다.
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
    # 📥 사전 학습 데이터(NPZ) 로드 (기존 유지)
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
    
    print("\n⏳ 언리얼 엔진이 폴더에 bin 파일을 생성하기를 기다리는 중...")
    os.makedirs(BIN_DIR_PATH, exist_ok=True)

    while True:
        try:
            bin_files = glob.glob(os.path.join(BIN_DIR_PATH, "*.bin"))
            
            if not bin_files:
                time.sleep(0.01)
                continue
                
            latest_file = max(bin_files, key=os.path.getmtime)
            mtime = os.path.getmtime(latest_file)
            fsize = os.path.getsize(latest_file)
            
            # 💡 [임시 수정] 중복 패스 로직을 주석 처리해서 계속 신호를 쏘게 만듭니다.
            # if latest_file == last_processed_file and mtime == last_mtime and fsize == last_size:
            #     time.sleep(0.005)
            #     continue
                
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
                        continue 
                    
                    action = agent.select_action(current_state)

                    # ... (생략: 윗부분 코드) ...
                    
                    if done_state != 0:
                        last_state = None
                        last_action = None
                        last_processed_file = latest_file
                        last_mtime = mtime
                        last_size = fsize
                        continue 
                    
                    # 💡 AI가 8개의 액션을 뽑아내는 바로 이 줄!
                    action = agent.select_action(current_state)
                    
                    # ==========================================
                    # 🔍 [이현수님 디버깅 코드 시작]
                    # ==========================================
                    try:
                        # 1. 만약 PyTorch Tensor나 NumPy 배열이라면 일반 파이썬 리스트로 변환
                        if hasattr(action, "detach"): # PyTorch용
                            debug_list = action.detach().cpu().numpy().tolist()
                        elif hasattr(action, "tolist"): # NumPy용
                            debug_list = action.tolist()
                        else:
                            debug_list = list(action) # 일반 이터러블용

                        # 2. 터미널(콘솔)에 보기 좋게 출력
                        import datetime
                        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        # 보기 편하게 소수점 4자리까지만 자름
                        formatted_list = [round(float(x), 4) for x in debug_list]
                        print(f"[{current_time}] [디버깅] 개수: {len(formatted_list)}개 | 데이터: {formatted_list}")

                    except Exception as e:
                        print(f"[디버깅 에러] 출력 실패: {e}")
                    # ==========================================
                    # 🔍 [이현수님 디버깅 코드 끝]
                    # ==========================================

                    last_state = current_state
                    last_action = action
                    
                    action_list = action.tolist() if isinstance(action, np.ndarray) else action
                    osc_client.send_message("/drone/input", action_list)
                    
                    # ... (생략: 아랫부분 코드) ...
                    
                    last_state = current_state
                    last_action = action
                    
                    action_list = action.tolist() if isinstance(action, np.ndarray) else action

# ==========================================
                    print("\n" + "="*40)
                    print(f"🧠 AI 모델 추론 완료! (Step: {int(current_step)})")
                    print("📊 [출력된 8개의 Action 값]")
                    for i, val in enumerate(action_list):
                        print(f"  ▶ Action [{i}] : {val:>8.4f}")
                    print("="*40 + "\n")
                    # ==========================================

                    osc_client.send_message("/drone/input", action_list)
                    
                    # 💡 연속 송신 확인 로그
                    print(f"🚀 [OSC 연속 송신 중...] 언리얼이 받을 때까지 쏘는 중 (파일: {os.path.basename(latest_file)})")
                    
            last_processed_file = latest_file
            last_mtime = mtime
            last_size = fsize
            
        except Exception as e:
            pass
            
        # 💡 과부하 방지를 위해 루프 속도를 0.05초로 변경
        time.sleep(0.05)

if __name__ == "__main__":
    run_server()