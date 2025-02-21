import pandas as pd
import json
import os

# 🔹 파일 경로
LOGDATA_JSON = "logdata.json"
ATTEMPTS_CSV = "user_attempts_log.csv"

# 🔹 `logdata.json`에서 사용자가 푼 모든 문제 가져오기 (맞은 문제 + 틀린 문제)
with open(LOGDATA_JSON, "r", encoding="utf-8") as f:
    logdata = json.load(f)

df = pd.DataFrame(logdata)
df['quiz_id'] = df['quiz_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)

# 🔹 사용자가 푼 모든 문제 필터링 (correct == 1 또는 correct == 0)
all_attempts = df[['user_id', 'quiz_id']]

# 🔹 기존 `user_attempts_log.csv` 불러오기 (없으면 빈 DataFrame 생성)
if os.path.exists(ATTEMPTS_CSV):
    attempts_log = pd.read_csv(ATTEMPTS_CSV, dtype=str)
else:
    attempts_log = pd.DataFrame(columns=['user_id', 'quiz_id'])

# 🔹 기존 기록과 새로운 데이터를 병합하여 중복 제거
updated_attempts = pd.concat([attempts_log, all_attempts]).drop_duplicates()

# 🔹 업데이트된 데이터를 CSV로 저장
updated_attempts.to_csv(ATTEMPTS_CSV, index=False)

print("✅ `user_attempts_log.csv`가 업데이트되었습니다! (사용자가 푼 모든 문제 저장됨)")
