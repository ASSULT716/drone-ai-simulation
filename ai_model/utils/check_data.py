import numpy as np
import os

def inspect_npz(file_path):
    print(f"🔍 데이터 검수를 시작합니다: {file_path}")
    
    if not os.path.exists(file_path):
        print("❌ 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    try:
        # 1. 파일 로드
        dataset = np.load(file_path)
        print("✅ 파일 로드 성공!\n")
        
        print("-" * 40)
        print("🔑 [포함된 데이터 명세서]")
        # 2. 파일 내부의 Key와 Shape 출력
        for key in dataset.files:
            data = dataset[key]
            print(f" 🔹 {key}: 형태 {data.shape} | 타입 {data.dtype}")
        print("-" * 40)

        # 3. 무결성 검사 (Sanity Check)
        print("\n🛡️ [데이터 무결성 검사]")
        if 'state' in dataset.files:
            state_data = dataset['state']
            if np.isnan(state_data).any() or np.isinf(state_data).any():
                print(" 🚨 [경고] state 배열에 빵꾸(NaN)나 무한대(Inf) 값이 있습니다! 팀원에게 수정을 요청하세요.")
            else:
                print(" ✅ state 배열 통과 (결측치 없음)")
        else:
            print(" ⚠️ 'state'라는 이름의 키가 없습니다. 팀원이 저장한 이름을 확인하세요.")

    except Exception as e:
        print(f"❌ 파일을 읽는 중 에러가 발생했습니다: {e}")

if __name__ == '__main__':
    # 📌 팀원에게 받은 npz 파일의 실제 경로를 여기에 적어주세요.
    # 예시: 상위 폴더의 data 폴더 안에 raw_data.npz 가 있다면
    TARGET_FILE = "../data/raw_data.npz" 
    
    inspect_npz(TARGET_FILE)