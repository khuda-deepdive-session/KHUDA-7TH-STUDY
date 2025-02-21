import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_model(model_path="lightfm_model.pkl"):
    """ 학습된 모델 로드 """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_model(model_path="lightfm_model.pkl"):
    """ 학습된 모델 로드 """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None

def recommend_items(model, dataset, user_id, item_features_df, n=5):
    """ 특정 user_id에 대한 추천 아이템과 예측 점수를 반환 + 각 점수 출력 """

    # 🔹 모델의 사용자 및 아이템 개수 가져오기
    num_users = model.user_embeddings.shape[0]
    num_items = model.item_embeddings.shape[0]

    user_ids = list(range(num_users))  # 사용자 인덱스 리스트
    item_ids = list(range(num_items))  # 아이템 인덱스 리스트

    # 🔹 dataset이 존재하는 경우, user_id를 숫자로 변환
    if dataset is not None:
        user_mapping = dataset.mapping()[0]  # 사용자 매핑 가져오기
        if user_id not in user_mapping:
            print(f"❌ 유저 {user_id}가 데이터셋에 존재하지 않습니다.")
            return []
        user_index = user_mapping[user_id]  # 🔥 user_id → 모델의 사용자 인덱스로 변환
    else:
        # 🔹 dataset이 없으면 `user_id`를 숫자로 변환
        try:
            user_index = int(user_id)  # 🔥 `user1`이 아니라 숫자(0~2)로 변환
        except ValueError:
            print(f"❌ 유저 {user_id}를 숫자로 변환할 수 없습니다.")
            return []

    if user_index is None or user_index >= num_users:
        print(f"❌ 유저 {user_id}가 모델에 존재하지 않습니다.")
        return []

    # 🔹 모델에서 기대하는 quiz_id 목록 가져오기
    trained_quiz_ids = set(range(num_items))

    # 🔹 현재 `item_features_df`의 quiz_id 목록 가져오기
    infer_quiz_ids = set(item_features_df['quiz_id'].unique())

    # 🔹 누락된 quiz_id 찾기
    missing_quiz_ids = trained_quiz_ids - infer_quiz_ids

    if missing_quiz_ids:
        print(f"⚠️ 누락된 quiz_id 개수: {len(missing_quiz_ids)}, 기본값(0)으로 채움.")
        missing_df = pd.DataFrame({'quiz_id': list(missing_quiz_ids)})
        missing_df['Reason_0_3'] = 0
        missing_df['Reason_K'] = 0
        item_features_df = pd.concat([item_features_df, missing_df], ignore_index=True)

    # 🔹 `quiz_id`를 정렬하여 모델과 순서를 맞춤
    item_features_df = item_features_df.sort_values(by="quiz_id").reset_index(drop=True)

    # 🔹 희소 행렬 변환
    item_features_csr = csr_matrix(item_features_df[['Reason_0_3', 'Reason_K']].values)

    # 🔹 LightFM 모델을 사용한 예측 수행
    scores = model.predict(user_index, np.arange(num_items), item_features=item_features_csr)

    # 🔹 점수 출력 (각 quiz_id별 예측 점수 확인)
    quiz_score_mapping = {item_ids[i]: scores[i] for i in range(len(scores))}
    sorted_quiz_scores = sorted(quiz_score_mapping.items(), key=lambda x: -x[1])

    print("\n🔹 [추천 점수 목록]")
    for quiz_id, score in sorted_quiz_scores[:n]:
        print(f"  - quiz_id {quiz_id}: 점수 {score:.4f}")

    # 🔹 점수가 높은 순서대로 정렬
    sorted_indices = np.argsort(-scores)[:n]
    recommended_items = [item_ids[i] for i in sorted_indices]

    return recommended_items
