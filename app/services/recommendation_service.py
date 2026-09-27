import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

class RecommendationService:
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @classmethod
    def collaborative_filtering(
        cls,
        ratings_data: List[Dict[str, Any]],
        target_user_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not ratings_data:
            return []

        df = pd.DataFrame(ratings_data)
        if "user_id" not in df.columns or "item_id" not in df.columns or "rating" not in df.columns:
            raise ValueError("Ratings data must contain user_id, item_id, and rating fields")

        pivot_df = df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0.0)

        if target_user_id not in pivot_df.index:
            mean_ratings = df.groupby("item_id")["rating"].mean().sort_values(ascending=False)
            results = []
            for item_id, score in mean_ratings.head(top_k).items():
                results.append({
                    "item_id": str(item_id),
                    "predicted_score": float(score),
                    "explanation": "Cold-start fallback: Popular item recommendation"
                })
            return results

        target_vector = pivot_df.loc[target_user_id].values
        user_ids = pivot_df.index.tolist()
        
        similarities = {}
        for uid in user_ids:
            if uid == target_user_id:
                continue
            other_vector = pivot_df.loc[uid].values
            sim = cls._cosine_similarity(target_vector, other_vector)
            if sim > 0:
                similarities[uid] = sim

        items = pivot_df.columns.tolist()
        unrated_items = [item for item in items if pivot_df.loc[target_user_id, item] == 0.0]

        predictions = []
        for item in unrated_items:
            weighted_sum = 0.0
            sim_sum = 0.0
            for uid, sim in similarities.items():
                rating = pivot_df.loc[uid, item]
                if rating > 0:
                    weighted_sum += sim * rating
                    sim_sum += sim

            predicted_rating = (weighted_sum / sim_sum) if sim_sum > 0 else 0.0
            predictions.append((item, predicted_rating))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recs = predictions[:top_k]

        return [
            {
                "item_id": str(item_id),
                "predicted_score": round(float(score), 4),
                "explanation": f"Collaborative filtering prediction based on {len(similarities)} similar users"
            }
            for item_id, score in top_recs
        ]

    @classmethod
    def content_based_filtering(
        cls,
        ratings_data: List[Dict[str, Any]],
        target_user_id: str,
        item_features: Dict[str, List[float]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        df = pd.DataFrame(ratings_data)
        user_ratings = df[df["user_id"] == target_user_id]
        
        if user_ratings.empty or not item_features:
            return []

        first_feature = next(iter(item_features.values()))
        feature_dim = len(first_feature)
        user_profile = np.zeros(feature_dim, dtype=np.float64)
        total_weight = 0.0

        rated_items = set()
        for _, row in user_ratings.iterrows():
            item_id = str(row["item_id"])
            rating = float(row["rating"])
            rated_items.add(item_id)
            if item_id in item_features:
                user_profile += rating * np.array(item_features[item_id], dtype=np.float64)
                total_weight += rating

        if total_weight > 0:
            user_profile /= total_weight

        predictions = []
        for item_id, feat_vec in item_features.items():
            if item_id in rated_items:
                continue
            feat_arr = np.array(feat_vec, dtype=np.float64)
            sim = cls._cosine_similarity(user_profile, feat_arr)
            predictions.append((item_id, sim))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_recs = predictions[:top_k]

        return [
            {
                "item_id": str(item_id),
                "predicted_score": round(float(score), 4),
                "explanation": "Content-based feature vector similarity matching user taste profile"
            }
            for item_id, score in top_recs
        ]
