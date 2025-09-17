import os
import joblib
import numpy as np
import tensorflow as tf

# -----------------------------
# Hybrid Recommender Class
# -----------------------------
class HybridRecommender:
    """
    HybridRecommender combines multiple recommendation strategies:
    - Collaborative Filtering (CF)
    - Content-based filtering (TF-IDF)
    - Visual recommendations (CNN embeddings)
    """

    def __init__(self, artifacts_path="artifacts"):
        print("Initializing the Hybrid Recommender...")
        self.artifacts_path = os.path.abspath(artifacts_path)

        # Model placeholders
        self.cf_model = None
        self.user_encoder = None
        self.article_encoder = None
        self.user_purchases = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix = None
        self.cnn_extractor = None
        self.image_embeddings = None
        self.image_paths = None
        self.visual_sim_matrix = None

        # Load all artifacts
        self._load_artifacts()

    # -----------------------------
    # Helper to get absolute path
    # -----------------------------
    def _full_path(self, filename):
        return os.path.join(self.artifacts_path, filename)

    # -----------------------------
    # Load all models and matrices
    # -----------------------------
    def _load_artifacts(self):
        print(f"Loading artifacts from {self.artifacts_path} ...")

        # Collaborative Filtering
        self.cf_model = tf.keras.models.load_model(self._full_path("collab_model.h5"))
        self.user_encoder = joblib.load(self._full_path("user_encoder.pkl"))
        self.article_encoder = joblib.load(self._full_path("article_encoder.pkl"))
        self.user_purchases = joblib.load(self._full_path("user_purchases.pkl"))

        # Content-based
        self.tfidf_vectorizer = joblib.load(self._full_path("tfidf_vectorizer.pkl"))
        self.tfidf_matrix = joblib.load(self._full_path("tfidf_matrix.pkl"))
        self.cosine_sim_matrix = joblib.load(self._full_path("cosine_sim_matrix.pkl"))

        # Visual recommender
        self.cnn_extractor = tf.keras.models.load_model(self._full_path("cnn_extractor.h5"))
        self.image_embeddings = np.load(self._full_path("image_embeddings.npy"))
        self.image_paths = joblib.load(self._full_path("image_paths.pkl"))
        self.visual_sim_matrix = joblib.load(self._full_path("visual_sim_matrix.pkl"))

        print("✅ All artifacts loaded successfully.")

    # -----------------------------
    # Fallback popular articles
    # -----------------------------
    def _get_popular_articles(self, n=10):
        # Simple most popular articles fallback
        return list(range(min(n, len(self.article_encoder.classes_))))

    # -----------------------------
    # Visual recommendations
    # -----------------------------
    def _get_visual_recs(self, article_id, n=10):
        if article_id not in self.article_encoder.classes_:
            return []
        idx = self.article_encoder.transform([article_id])[0]
        sim_scores = self.visual_sim_matrix[idx]
        top_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        return self.article_encoder.inverse_transform(top_indices)

    # -----------------------------
    # Content-based recommendations
    # -----------------------------
    def _get_content_recs(self, article_id, n=10):
        if article_id not in self.article_encoder.classes_:
            return []
        idx = self.article_encoder.transform([article_id])[0]
        sim_scores = self.cosine_sim_matrix[idx]
        top_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        return self.article_encoder.inverse_transform(top_indices)

    # -----------------------------
    # Collaborative Filtering recommendations (batch safe)
    # -----------------------------
    def _get_collaborative_recs(self, user_id, n=10, batch_size=500):
        if user_id not in self.user_encoder.classes_:
            return self._get_popular_articles(n)

        user_code = self.user_encoder.transform([user_id])[0]
        purchased_articles = self.user_purchases.get(user_code, set())

        # Restrict candidate articles to CF model's max index
        max_article_idx = self.cf_model.get_layer('article_embedding').input_dim - 1
        candidate_articles = [
            i for i in range(max_article_idx + 1) if i not in purchased_articles
        ]

        if not candidate_articles:
            return self._get_popular_articles(n)

        # Batch prediction to prevent freezing
        preds = []
        for start in range(0, len(candidate_articles), batch_size):
            end = start + batch_size
            batch_articles = candidate_articles[start:end]
            X_user = np.array([user_code] * len(batch_articles))
            X_article = np.array(batch_articles)
            batch_preds = self.cf_model.predict([X_user, X_article], verbose=0).flatten()
            preds.extend(batch_preds)

        preds = np.array(preds)
        top_indices = preds.argsort()[::-1][:n]
        return self.article_encoder.inverse_transform(np.array(candidate_articles)[top_indices])

    # -----------------------------
    # Hybrid recommendations
    # -----------------------------
    def get_recommendations(self, user_id, article_id, n=12):
        visual_recs = self._get_visual_recs(article_id)
        content_recs = self._get_content_recs(article_id)
        collaborative_recs = self._get_collaborative_recs(user_id)

        # Weighted scoring
        scores = {}
        for rec in visual_recs:
            scores[rec] = scores.get(rec, 0) + 3
        for rec in collaborative_recs:
            scores[rec] = scores.get(rec, 0) + 2
        for rec in content_recs:
            scores[rec] = scores.get(rec, 0) + 1

        # Sort by score
        final_recs = [rec for rec, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        final_recs = [r for r in final_recs if r != article_id]
        return final_recs[:n]
