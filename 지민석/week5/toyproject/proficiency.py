import pandas as pd
import json
import os

# 🔹 CSV 파일 경로
LOGDATA_JSON = "logdata.json"
PROFICIENCY_CSV = "user_proficiency.csv"
MODIFIED_DATA_CSV = "modified_data.csv"

# 🔹 JSON 파일 불러오기
with open(LOGDATA_JSON, "r", encoding="utf-8") as f:
    logdata = json.load(f)

# 🔹 pandas DataFrame 변환
df = pd.DataFrame(logdata)
df['quiz_id'] = df['quiz_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)

# 🔹 문제 데이터 불러오기
data_df = pd.read_csv(MODIFIED_DATA_CSV, encoding="utf-8")
data_df['Question'] = data_df['Question'].astype(str)  # 🔥 `quiz_id`와 같은 타입(str)으로 변환
reason_mapping = data_df.set_index('Question')[['Reason_0_3', 'Reason_K']].to_dict(orient='index')

# 🔹 사용자 숙련도 불러오기 (없으면 새로 생성)
if os.path.exists(PROFICIENCY_CSV) and os.stat(PROFICIENCY_CSV).st_size > 0:
    user_proficiency = pd.read_csv(PROFICIENCY_CSV).set_index('user_id').to_dict(orient='index')
else:
    # 🔹 `Reason_0_3_0` ~ `Reason_0_3_3` (4개) + `Reason_K_0` ~ `Reason_K_12` (13개) 총 17개 컬럼 초기화
    user_proficiency = {}

# 🔹 초기화할 컬럼 목록 생성
all_columns = [f"Reason_0_3_{i}" for i in range(4)] + [f"Reason_K_{i}" for i in range(13)]

def update_user_proficiency(user_id, quiz_id, correct):
    """ 사용자가 특정 문제를 풀었을 때 숙련도를 업데이트 """
    if quiz_id in reason_mapping:
        try:
            reason_0_3_val = int(reason_mapping[quiz_id]['Reason_0_3'])  # 🔥 정수 변환 후 사용
            reason_K_val = int(reason_mapping[quiz_id]['Reason_K'])  # 🔥 정수 변환 후 사용
            reason_0_3 = f"Reason_0_3_{reason_0_3_val}"
            reason_K = f"Reason_K_{reason_K_val}"

            print(f"🔹 업데이트 중: user_id={user_id}, quiz_id={quiz_id}, correct={correct}")
            print(f"   → Reason_0_3: {reason_0_3}, Reason_K: {reason_K}")

            # 🔹 사용자 ID가 없으면 모든 컬럼을 100으로 초기화
            if user_id not in user_proficiency:
                user_proficiency[user_id] = {col: 100 for col in all_columns}
                print(f"   → 사용자 {user_id} 추가됨. (숙련도 100으로 초기화 완료)")

            # 🔹 숙련도 업데이트 
            if correct == 1:
                user_proficiency[user_id][reason_0_3] += 1
                user_proficiency[user_id][reason_K] += 1
            else:
                user_proficiency[user_id][reason_0_3] -= 0.5  # 틀린 경우 가중치 증가
                user_proficiency[user_id][reason_K] -= 0.5

            print(f"   → 숙련도 업데이트 완료: {user_proficiency[user_id]}")

        except KeyError:
            print(f"🚨 오류: quiz_id {quiz_id}에 대한 Reason_0_3 또는 Reason_K 데이터가 없습니다.")

# 🔹 `logdata.json`을 기반으로 숙련도 행렬 업데이트
for _, row in df.iterrows():
    update_user_proficiency(row['user_id'], row['quiz_id'], row['correct'])

# 🔹 업데이트된 숙련도를 DataFrame으로 변환 및 CSV로 저장
df_proficiency = pd.DataFrame.from_dict(user_proficiency, orient='index').reset_index().rename(columns={'index': 'user_id'})

# 🔹 NaN 값이 존재하는 경우 100으로 대체
df_proficiency.fillna(100, inplace=True)

df_proficiency.to_csv(PROFICIENCY_CSV, index=False)

print("✅ 사용자별 숙련도 행렬이 `user_proficiency.csv`에 저장됨!")
