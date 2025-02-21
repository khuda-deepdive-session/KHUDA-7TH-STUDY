import pickle
import numpy as np
import pandas as pd
from lightfm import LightFM
import os

# 🔹 저장된 모델 및 데이터셋 로드
model_path = "lightfm_model.pkl"
dataset_path = "dataset.pkl"
attempts_log_path = "user_attempts_log.csv"
proficiency_path = "user_proficiency.csv"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)  # 🔥 저장된 dataset 로드

# 🔹 문제 데이터 불러오기
data_df = pd.read_csv("modified_data.csv", encoding="utf-8")
item_features_df = data_df[['Question', 'Reason_0_3', 'Reason_K']].drop_duplicates()
item_features_df = item_features_df.rename(columns={'Question': 'quiz_id'})
item_features_df['quiz_id'] = item_features_df['quiz_id'].astype(str)  # 🔥 quiz_id를 문자열로 변환

# 🔹 사용자 푼 문제 기록 불러오기 (없으면 빈 딕셔너리)
if os.path.exists(attempts_log_path):
    user_attempts_log = pd.read_csv(attempts_log_path, dtype=str).groupby('user_id')['quiz_id'].apply(set).to_dict()
else:
    user_attempts_log = {}

# 🔹 사용자 숙련도 불러오기 (없으면 빈 딕셔너리)
if os.path.exists(proficiency_path):
    user_proficiency = pd.read_csv(proficiency_path).set_index('user_id').to_dict(orient='index')
else:
    user_proficiency = {}

# 🔹 추천 함수
def recommend_items(model, dataset, user_id, item_features_df, n=5):
    """ 특정 user_id에 대한 추천 아이템 반환 (이미 푼 문제 제외, 숙련도 반영) """
    num_users = model.user_embeddings.shape[0]
    num_items = model.item_embeddings.shape[0]

    user_mapping = dataset.mapping()[0]  # 🔹 user_id 매핑 정보
    item_mapping = dataset.mapping()[2]  # 🔹 quiz_id 매핑 정보

    if user_id not in user_mapping:
        print(f"❌ 유저 {user_id}가 데이터셋에 존재하지 않습니다.")
        return []

    user_index = user_mapping[user_id]  # 🔥 user_id를 모델 인덱스로 변환
    scores = model.predict(user_index, np.arange(num_items))

    # 🔹 숙련도 기반으로 추천 점수 조정
    if user_id in user_proficiency:
        proficiency_data = user_proficiency[user_id]

        for i, quiz_id in enumerate(item_mapping.keys()):
            quiz_data = item_features_df[item_features_df['quiz_id'] == quiz_id]

            if quiz_data.empty:
                continue

            try:
                reason_0_3_key = f"Reason_0_3_{int(quiz_data['Reason_0_3'].values[0])}"
                reason_K_key = f"Reason_K_{int(quiz_data['Reason_K'].values[0])}"

                # 🔹 숙련도가 높을수록 점수 감소, 낮을수록 점수 증가
                proficiency_bonus = (100 - proficiency_data.get(reason_0_3_key, 100)) * 0.05 + \
                    (100 - proficiency_data.get(reason_K_key, 100)) * 0.05


                scores[i] += proficiency_bonus  # 🔥 숙련도가 낮을수록 점수 증가, 높을수록 감소

                # 🔍 `quiz_id = 6`의 숙련도 보정 점수 확인
                if quiz_id == "6":
                    print(f"🔎 `quiz_id = 6` 숙련도 보정 점수: {proficiency_bonus:.4f}, "
                          f"{reason_0_3_key} = {proficiency_data.get(reason_0_3_key, 100)}, "
                          f"{reason_K_key} = {proficiency_data.get(reason_K_key, 100)}")

            except (KeyError, ValueError):
                print(f"🚨 오류: quiz_id {quiz_id}에 대한 Reason_0_3 또는 Reason_K 값을 찾을 수 없습니다.")

    # 🔍 LightFM 기본 점수 출력
    quiz_scores = {quiz_id: scores[i] for i, quiz_id in enumerate(item_mapping.keys())}
    sorted_scores = sorted(quiz_scores.items(), key=lambda x: -x[1])  # 점수 높은 순으로 정렬

    print("🔍 전체 문제 추천 점수 (상위 10개):")
    for quiz_id, score in sorted_scores[:10]:
        print(f"   - quiz_id: {quiz_id}, 추천 점수: {score:.4f}")

    # 🔹 특정 `quiz_id = 6`의 점수 확인
    if "6" in quiz_scores:
        print(f"🔎 `quiz_id = 6` 추천 점수: {quiz_scores['6']:.4f}")

    # 🔹 점수 기준으로 상위 추천 후보 선정
    recommended_items = np.argsort(-scores)

    # 🔹 역매핑을 사용하여 실제 `quiz_id` 찾기
    reverse_item_mapping = {v: str(k) for k, v in item_mapping.items()}  # 🔥 역매핑 생성 (quiz_id를 문자열로 변환)
    recommended_quiz_ids = [reverse_item_mapping[i] for i in recommended_items if i in reverse_item_mapping]

    # 🔹 사용자가 이미 푼 문제 필터링
    if user_id in user_attempts_log:
        solved_problems = set(str(qid) for qid in user_attempts_log[user_id])  # 🔥 모든 quiz_id를 문자열로 변환하여 비교
        recommended_quiz_ids = [quiz for quiz in recommended_quiz_ids if quiz not in solved_problems]

    # 🔹 추천 개수가 부족하면 추가 문제 추천
    if len(recommended_quiz_ids) < n:
        print(f"⚠️ 추천 개수가 부족함: {len(recommended_quiz_ids)}/{n}")
        remaining_items = set(map(str, item_mapping.keys())) - set(recommended_quiz_ids)
        additional_recommendations = list(remaining_items)[:n - len(recommended_quiz_ids)]
        recommended_quiz_ids.extend(additional_recommendations)

    return recommended_quiz_ids[:n]

# 🔹 예제: user_id "user1"에게 추천 실행
user_id = "user1"
recommended = recommend_items(model, dataset, user_id, item_features_df, n=5)
print(f"🎯 {user_id}에게 추천되는 문제: {recommended}")
