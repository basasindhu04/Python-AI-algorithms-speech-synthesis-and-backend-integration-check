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
        """
        User-Based Collaborative Filtering using Pandas DataFrame & NumPy Cosine Similarity.
        """
        if not ratings_data:
            return []

        df = pd.DataFrame(ratings_data)
        if "user_id" not in df.columns or "item_id" not in df.columns or "rating" not in df.columns:
            raise ValueError("Ratings data must contain user_id, item_id, and rating fields")

        # Create User-Item Pivot Matrix
        pivot_df = df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0.0)

        if target_user_id not in pivot_df.index:
            # Cold-start fallback: return globally top-rated items
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
        
        # Calculate User-User Similarities
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
        """
        Content-Based Filtering: Build user profile vector from rated item features,
        and score unrated items using cosine similarity.
        """
        df = pd.DataFrame(ratings_data)
        user_ratings = df[df["user_id"] == target_user_id]
        
        if user_ratings.empty or not item_features:
            return []

        # Construct user profile as weighted average of item feature vectors
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
                continue  # Skip already rated
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

    @classmethod
    def get_recommendations(
        cls,
        ratings_data: List[Dict[str, Any]],
        target_user_id: str,
        strategy: str = "collaborative",
        item_features: Optional[Dict[str, List[float]]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()

        if strategy == "content_based":
            if not item_features:
                raise ValueError("item_features required for content_based strategy")
            recs = cls.content_based_filtering(ratings_data, target_user_id, item_features, top_k)
        elif strategy == "collaborative":
            recs = cls.collaborative_filtering(ratings_data, target_user_id, top_k)
        elif strategy == "hybrid":
            collab_recs = {r["item_id"]: r["predicted_score"] for r in cls.collaborative_filtering(ratings_data, target_user_id, top_k*2)}
            content_recs = {r["item_id"]: r["predicted_score"] for r in cls.content_based_filtering(ratings_data, target_user_id, item_features or {}, top_k*2)}
            
            all_items = set(collab_recs.keys()).union(set(content_recs.keys()))
            hybrid_scores = []
            for item_id in all_items:
                c_score = collab_recs.get(item_id, 0.0)
                cnt_score = content_recs.get(item_id, 0.0)
                # Hybrid score: 60% collaborative + 40% content
                combined = 0.6 * c_score + 0.4 * cnt_score
                hybrid_scores.append((item_id, combined))

            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            recs = [
                {
                    "item_id": str(item_id),
                    "predicted_score": round(float(score), 4),
                    "explanation": "Hybrid weighted combination (60% Collaborative + 40% Content-Based)"
                }
                for item_id, score in hybrid_scores[:top_k]
            ]
        else:
            raise ValueError(f"Unknown recommendation strategy: {strategy}")

        end_time = time.perf_counter()
        return {
            "target_user_id": target_user_id,
            "strategy_used": strategy,
            "recommendations": recs,
            "execution_time_ms": (end_time - start_time) * 1000
        }
