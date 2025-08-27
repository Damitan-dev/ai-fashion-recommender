import joblib  # Efficient library to save/load Python objects like encoders, lists, or matrices
import numpy as np  # For numerical computations, arrays, and matrix operations
import tensorflow as tf  # Deep learning library for loading Keras models
import os

# -----------------------------
# Define a Hybrid Recommender Class
# -----------------------------
class HybridRecommender:
    """
    HybridRecommender combines multiple recommendation strategies:
    - Collaborative Filtering (CF)
    - Content-based filtering (e.g., TF-IDF)
    - Visual recommendations (CNN embeddings)
    
    Loads all artifacts (pre-trained models, matrices, encoders) and provides
    a single method to generate recommendations.
    """

    def __init__(self, artifacts_path="artifacts"):
        """
        Initializes the Hybrid Recommender by loading all models and artifacts from disk.
        Uses absolute paths so it works regardless of where the script is run from.
        """
        print("Initializing the Hybrid Recommender...")

        # Resolve artifacts directory to absolute path
        self.artifacts_path = os.path.abspath(artifacts_path)

        # --- Attributes for loaded objects ---
        self.cf_model = None
        self.user_encoder = None
        self.article_encoder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim_matrix = None
        self.cnn_extractor = None
        self.image_embeddings = None
        self.image_paths = None
        self.visual_sim_matrix = None
        self.user_purchases = None

        # Load everything
        self._load_artifacts()

    def _full_path(self, filename):
        """Helper: safely join filename with artifacts path"""
        return os.path.join(self.artifacts_path, filename)

    def _load_artifacts(self):
        """
        Loads all saved artifacts (models, encoders, matrices) into memory.
        """
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

    def _get_visual_recs(self, article_id, n=10):
        """Returns top N visually similar articles using precomputed similarity matrix."""
        idx = self.article_encoder.transform([article_id])[0]
        sim_scores = self.visual_sim_matrix[idx]
        top_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        return self.article_encoder.inverse_transform(top_indices)

    def _get_content_recs(self, article_id, n=10):
        """Returns top N content-based similar articles using TF-IDF."""
        idx = self.article_encoder.transform([article_id])[0]
        sim_scores = self.cosine_sim_matrix[idx]
        top_indices = sim_scores.argsort()[::-1]
        top_indices = [i for i in top_indices if i != idx][:n]
        return self.article_encoder.inverse_transform(top_indices)

    def _get_collaborative_recs(self, user_id, n=10):
        """Returns top N personalized articles for a given user using collaborative filtering."""
        user_code = self.user_encoder.transform([user_id])[0]
        purchased_articles = self.user_purchases.get(user_code, set())
        candidate_articles = [i for i in range(len(self.article_encoder.classes_)) if i not in purchased_articles]

        X_user = np.array([user_code] * len(candidate_articles))
        X_article = np.array(candidate_articles)
        preds = self.cf_model.predict([X_user, X_article]).flatten()

        top_indices = preds.argsort()[::-1][:n]
        return self.article_encoder.inverse_transform(np.array(candidate_articles)[top_indices])

    def get_recommendations(self, user_id, article_id, n=12):
        """
        Generate hybrid recommendations by combining visual, content, and collaborative signals.
        """
        print(f"Generating recommendations for user {user_id} and article {article_id}...")

        # Get recommendations from all models
        visual_recs = self._get_visual_recs(article_id)
        content_recs = self._get_content_recs(article_id)
        collaborative_recs = self._get_collaborative_recs(user_id)

        # Weighted scoring
        recommendation_scores = {}
        for rec_id in visual_recs:
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 3
        for rec_id in collaborative_recs:
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 2
        for rec_id in content_recs:
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 1

        sorted_recs = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
        final_recs = [rec_id for rec_id, score in sorted_recs if rec_id != article_id]
        return final_recs[:n]