import pandas as pd

# CSV 파일 로드
file_path = "data.csv"
df = pd.read_csv(file_path)

# "Question" 열에서 숫자만 추출하여 정수형으로 변환
df['Question'] = df['Question'].str.extract('(\d+)').astype(int)

# ✅ `Reason_0_3`과 `Reason_K`에 숫자가 포함된 행만 유지
df = df[df['Reason_0_3'].str.contains(r'\d', na=False)]
df = df[df['Reason_K'].str.contains(r'\d', na=False)]

# ✅ 모든 문자열 값에서 큰따옴표 (") 제거
df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)

# 데이터 저장
modified_file_path = "modified_data.csv"
df.to_csv(modified_file_path, index=False)

print("✅ 데이터 변환 완료: `Reason_0_3`과 `Reason_K`에 숫자가 전혀 없는 행이 삭제되었으며, 모든 큰따옴표(\")가 제거되었습니다.")
