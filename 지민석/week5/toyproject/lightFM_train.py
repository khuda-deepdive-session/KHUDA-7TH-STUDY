import pandas as pd
import os
import json
import pickle
from lightfm import LightFM
from lightfm.data import Dataset

# 🔹 CSV 파일 경로
LOGDATA_JSON = "logdata.json"
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
item_features_df = data_df[['Question', 'Reason_0_3', 'Reason_K']].drop_duplicates()
item_features_df = item_features_df.rename(columns={'Question': 'quiz_id'})
item_features_df['quiz_id'] = item_features_df['quiz_id'].astype(str)

# 🔹 LightFM Dataset 생성
dataset = Dataset()
all_quiz_ids = item_features_df['quiz_id'].unique()

dataset.fit(
    users=df['user_id'].unique(),
    items=all_quiz_ids,
    item_features=[f"Reason_0_3_{x}" for x in item_features_df['Reason_0_3'].unique()] +
                  [f"Reason_K_{x}" for x in item_features_df['Reason_K'].unique()]
)

print(f"🔥 dataset.fit 실행 후 quiz_id 개수: {len(dataset.mapping()[2])}")

# ✅ `dataset.fit_partial()` 실행 전에 누락된 quiz_id 보완
dataset.fit_partial(items=all_quiz_ids)

print(f"🔥 dataset.fit_partial 실행 후 quiz_id 개수: {len(dataset.mapping()[2])}")

# 🔹 사용자가 푼 문제 목록
user_attempts = df.groupby('user_id')['quiz_id'].apply(set).to_dict()

# 🔹 Negative 샘플 추가
all_quiz_ids_set = set(all_quiz_ids)
negative_samples = []

for user, solved in user_attempts.items():
    unsolved = list(all_quiz_ids_set - solved)  # 사용자가 풀지 않은 문제
    for quiz_id in unsolved[:5]:  # 각 사용자당 최대 5개의 Negative 샘플 추가
        negative_samples.append((user, quiz_id, 0.1))

# 🔹 기존 데이터 + Negative 샘플 포함
interaction_data = [(row['user_id'], row['quiz_id'], max(0.1, 1.0 if row['correct'] == 1 else 0.5)) for _, row in df.iterrows()]
interaction_data.extend(negative_samples)

# ✅ 유저-문제 상호작용 행렬 구축 (Negative 샘플 포함)
(interactions, weights) = dataset.build_interactions(interaction_data)

# ✅ 아이템 Feature 행렬 구축 (형식 변환)
item_features = dataset.build_item_features(
    ((row['quiz_id'], [f"Reason_0_3_{row['Reason_0_3']}", f"Reason_K_{row['Reason_K']}"]) for _, row in item_features_df.iterrows())
)

print("🔹 아이템 Feature 매핑 완료")

# ✅ LightFM 모델 학습 (Negative 샘플 포함)
model = LightFM(loss='logistic')
model.fit(interactions, item_features=item_features, sample_weight=weights, epochs=10, num_threads=8)

# ✅ 모델 저장
model_path = "lightfm_model.pkl"
dataset_path = "dataset.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(dataset_path, "wb") as f:
    pickle.dump(dataset, f)  # 🔥 dataset 저장

print(f"✅ 모델이 저장됨: {model_path}")
print(f"✅ 데이터셋이 저장됨: {dataset_path}")
print("✅ LightFM 모델 학습 완료 및 저장!")
