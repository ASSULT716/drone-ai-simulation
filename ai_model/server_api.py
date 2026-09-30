import os
import glob
import time
import struct
import numpy as np
from pythonosc import udp_client

class UnrealCommunicationAPI:
    def __init__(self, bin_dir, osc_ip="100.82.247.96", osc_port=7000):
        self.bin_dir = bin_dir
        self.osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
        self.state_dim = 62  # 전체 상태 차원 (Step, Done, Reward 제외한 센서/관측 62차원)
        self.packet_size = 248  # 62 float * 4 bytes = 248 bytes
        self.last_processed_file = None
        self.last_fsize = 0
        self.episode_start_time = None
        self.engine_started = False
        self.backspace_hold_until = 0.0

        os.makedirs(self.bin_dir, exist_ok=True)

    def clear_old_bins(self):
        """서버 시작 시 남아있는 잔여 .bin 파일 정리"""
        old_files = glob.glob(os.path.join(self.bin_dir, "*.bin"))
        for f in old_files:
            try:
                os.remove(f)
            except Exception:
                pass

    def reset_file_pointer(self):
        """에피소드 종료 시 타이머 및 상태 초기화"""
        self.episode_start_time = None
        self.engine_started = False
        self.backspace_hold_until = 0.0
        self.last_processed_file = None
        self.last_fsize = 0

    def read_next_state(self, timeout=0.05):
        """
        언리얼 엔진의 최신 .bin 파일을 찾아 최신 62차원 상태를 읽어옴.
        반환: state, reward, done, is_valid
        """
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            bin_files = glob.glob(os.path.join(self.bin_dir, "*.bin"))
            if not bin_files:
                time.sleep(0.005)
                continue

            latest_file = max(bin_files, key=os.path.getmtime)
            fsize = os.path.getsize(latest_file)

            if fsize < self.packet_size:
                time.sleep(0.005)
                continue

            # 파일 크기에 변화가 있거나 새로운 파일인 경우 처리
            if latest_file != self.last_processed_file or fsize > self.last_fsize:
                with open(latest_file, "rb") as f:
                    f.seek(fsize - self.packet_size)
                    data = f.read(self.packet_size)

                if len(data) == self.packet_size:
                    # 62개의 float 언팩
                    state_tuple = struct.unpack('<62f', data)
                    raw_data = np.array(state_tuple, dtype=np.float32)

                    # 상태 정규화: 1000.0 나누기 및 [-1.0, 1.0] 클리핑
                    state = np.clip(raw_data / 1000.0, -1.0, 1.0)
                    
                    # 더미 보상 및 done 설정 (필요시 패킷 헤더/포맷에 맞춰 분리 가능)
                    reward = 0.1 
                    done = False

                    self.last_processed_file = latest_file
                    self.last_fsize = fsize

                    if self.episode_start_time is None:
                        self.episode_start_time = time.time()

                    return state, reward, done, True

            time.sleep(0.005)

        return np.zeros(self.state_dim, dtype=np.float32), 0.0, False, False

    def send_action(self, raw_action, done=False):
        """
        8차원 액션을 언리얼 엔진 OSC 규격에 맞게 변환하여 발신
        """
        processed_action = np.zeros(8, dtype=np.float32)

        # 1~4: Move Up, Right, Forward, Input Roll (-1, 0, 1)
        processed_action[0] = np.round(float(raw_action[0]))
        processed_action[1] = np.round(float(raw_action[1]))
        processed_action[2] = 1.0 if raw_action[2] > 0.0 else np.round(float(raw_action[2]))
        processed_action[3] = np.round(float(raw_action[3]))

        # 5~6: Input Look Up, Input Turn (-1.0 ~ 1.0)
        processed_action[4] = float(raw_action[4])
        processed_action[5] = float(raw_action[5])

        # 7: Pressing_I (엔진 시동: 1.6초 후 1회 트리거)
        current_sys_time = time.time()
        elapsed_time = current_sys_time - self.episode_start_time if self.episode_start_time else 0.0
        if elapsed_time >= 1.6 and not self.engine_started:
            processed_action[6] = 1.0
            self.engine_started = True
        else:
            processed_action[6] = 0.0

        # 8: Is_Pressing_Backspace (리트라이 트리거 유지)
        if raw_action[7] > 0 or done:
            self.backspace_hold_until = current_sys_time + 2.5

        if current_sys_time <= self.backspace_hold_until:
            processed_action[7] = 1.0
        else: 
            processed_action[7] = 0.0

        action_list = [float(x) for x in processed_action.tolist()]
        self.osc_client.send_message("/drone/input", action_list)