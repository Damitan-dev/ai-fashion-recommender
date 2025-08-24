import joblib  # Efficient library to save/load Python objects like encoders, lists, or matrices
import numpy as np  # For numerical computations, arrays, and matrix operations
import tensorflow as tf  # Deep learning library for loading Keras models


# -----------------------------
# Define a Hybrid Recommender Class
# -----------------------------
class HybridRecommender:
    """
    HybridRecommender combines multiple recommendation strategies:
    - Collaborative Filtering (CF)
    - Content-based filtering (e.g., TF-IDF)
    - Visual recommendations (CNN embeddings)
    
    The class loads all artifacts (pre-trained models, matrices, encoders) and provides
    a single method to generate recommendations.
    
    Think of this class as a team manager:
    - It gathers all experts (models, encoders, embeddings)
    - It knows how to ask each expert to give their suggestions
    - It merges their advice into a final recommendation list
    """
    
    def __init__(self, artifacts_path='../artifacts/'):
        """
        Initialization method that runs as soon as you create a new HybridRecommender object.
        Loads all models and artifacts from disk for use.
        """
        print("Initializing the Hybrid Recommender...")
        
        # Path to the directory containing all your saved artifacts
        self.artifacts_path = artifacts_path
        
        # --- Attributes for loaded objects ---
        # These will store all your models, encoders, and matrices after loading
        self.cf_model = None            # Collaborative filtering model (trained on user-item interactions)
        self.user_encoder = None        # Maps real user IDs to integer codes the CF model understands
        self.article_encoder = None     # Maps article IDs to integer codes
        self.tfidf_vectorizer = None   # TF-IDF vectorizer for content-based recommendations
        self.tfidf_matrix = None       # Precomputed TF-IDF matrix
        self.cosine_sim_matrix = None  # Precomputed cosine similarity for content-based similarity
        self.cnn_extractor = None      # CNN feature extractor model
        self.image_embeddings = None   # Precomputed visual embeddings for images
        self.image_paths = None        # Paths to images corresponding to embeddings
        self.visual_sim_matrix = None  # Precomputed cosine similarity for images
        self.user_purchases = None
        # Load all artifacts immediately when class is instantiated
        self._load_artifacts()



    def _load_artifacts(self):
        """
        This private method loads all saved artifacts (models, encoders, matrices) into memory.
        Why private? By convention, we don’t want users to call this directly; it’s internal setup.
        
        Think of this as onboarding new team members before starting a project: you bring in all experts
        so they’re ready when you need their advice.
        """
        print("Loading artifacts...")

        # --- Example of loading each artifact ---
        # Load CF model
        self.cf_model = tf.keras.models.load_model(self.artifacts_path + 'collab_model.h5')
        # Load user and article encoders
        self.user_encoder = joblib.load(self.artifacts_path + 'user_encoder.pkl')
        self.article_encoder = joblib.load(self.artifacts_path + 'article_encoder.pkl')
        self.user_purchases = joblib.load(self.artifacts_path + 'user_purchases.pkl')

        # Load content-based artifacts
        self.tfidf_vectorizer = joblib.load(self.artifacts_path + 'tfidf_vectorizer.pkl')
        self.tfidf_matrix = joblib.load(self.artifacts_path + 'tfidf_matrix.pkl')
        self.cosine_sim_matrix = joblib.load(self.artifacts_path + 'cosine_sim_matrix.pkl')
        # Load visual recommender artifacts
        self.cnn_extractor = tf.keras.models.load_model(self.artifacts_path + 'cnn_extractor.h5')
        self.image_embeddings = np.load(self.artifacts_path + 'image_embeddings.npy')
        self.image_paths = joblib.load(self.artifacts_path + 'image_paths.pkl')
        # NEW: load precomputed visual similarity
        self.visual_sim_matrix = joblib.load(self.artifacts_path + 'visual_sim_matrix.pkl')

        print("All artifacts loaded successfully.")


    def _get_visual_recs(self, article_id, n=10):
        """
        Returns top N visually similar articles using the precomputed similarity matrix.
        """

        # 1. Convert the external article ID to the internal integer code used by the encoders
        #    This is necessary because both the embeddings and similarity matrix are indexed by integer codes.
        idx = self.article_encoder.transform([article_id])[0]

        # 2. Retrieve the similarity scores for this article from the precomputed visual similarity matrix
        #    self.visual_sim_matrix[idx] is a 1D array where each element represents the similarity
        #    between this article and every other article in the dataset
        sim_scores = self.visual_sim_matrix[idx]

        # 3. Sort the indices of the similarity scores in descending order (highest similarity first)
        #    argsort returns indices that would sort the array; [::-1] reverses it to get descending order
        top_indices = sim_scores.argsort()[::-1]      

        # 4. Filter out the article itself (we don’t want to recommend the same article as similar)
        #    Then take the top N most similar articles
        top_indices = [i for i in top_indices if i != idx][:n]  

        # 5. Convert the internal integer codes back to the actual article IDs
        #    This ensures the method returns meaningful IDs that can be used elsewhere in the recommender
        return self.article_encoder.inverse_transform(top_indices)


        

    def _get_content_recs(self, article_id, n=10):
        """
        Returns top N content-based similar articles using TF-IDF.
        This method works even if you forgot to import cosine_similarity elsewhere,
        because we precomputed the similarity matrix when building the recommender.
        """
        # 1. Convert the article_id to its internal integer code
        idx = self.article_encoder.transform([article_id])[0]
        
        # 2. Retrieve precomputed similarity scores for this article
        #    Each entry in self.cosine_sim_matrix[idx] corresponds to similarity with another article
        sim_scores = self.cosine_sim_matrix[idx]
        
        # 3. Sort indices by similarity score in descending order
        #    Exclude the article itself (idx) and keep top N
        top_indices = sim_scores.argsort()[::-1]        # Sort descending
        top_indices = [i for i in top_indices if i != idx][:n]
        
        # 4. Convert internal codes back to actual article IDs
        return self.article_encoder.inverse_transform(top_indices)



        



    def _get_collaborative_recs(self, user_id, n=10):
        """
        Returns the top N personalized articles for a given user using the collaborative filtering (CF) model.
        This method predicts which articles the user is most likely to purchase next.
        """
        
        # 1. Encode the user ID into an integer code that the CF model understands
        #    CF models work with integer indices, not raw user IDs. 
        #    This step maps your real user ID to the internal code used during training.
        user_code = self.user_encoder.transform([user_id])[0]
        
        # 2. Retrieve the set of articles the user has already purchased
        #    We don’t want to recommend items they already bought.
        purchased_articles = self.user_purchases.get(user_code, set())
        
        # 3. Identify candidate articles the user has not purchased yet
        #    These are the items the model can score to make recommendations.
        candidate_articles = [i for i in range(len(self.article_encoder.classes_)) if i not in purchased_articles]
        
        # 4. Prepare model input arrays
        #    - Repeat the user_code for each candidate article
        #    - This lets the CF model predict a probability for each candidate article
        X_user = np.array([user_code]*len(candidate_articles))
        X_article = np.array(candidate_articles)
        
        # 5. Predict the purchase probability for each candidate article
        #    The model outputs a number between 0-1 indicating how likely the user is to purchase.
        preds = self.cf_model.predict([X_user, X_article]).flatten()
        
        # 6. Select the indices of the top N predictions
        #    argsort()[::-1] sorts from highest to lowest probability
        #    [:n] takes only the top N items
        top_indices = preds.argsort()[::-1][:n]
        
        # 7. Convert the internal article indices back to the original article IDs
        #    This makes the recommendations human-readable / usable in real applications
        return self.article_encoder.inverse_transform(np.array(candidate_articles)[top_indices])



    def get_recommendations(self, user_id, article_id, n=12):
        """
        Public method to generate hybrid recommendations for a specific user & article.
        Placeholder for now; tomorrow we’ll merge content, collaborative, and visual recs.
        
        Think of it like calling your team of experts:
        - CF expert gives personalized suggestions
        - Content expert suggests similar products
        - Visual expert suggests similar-looking items
        - You merge their outputs into one final list
        """
        print(f"Generating recommendations for user {user_id} and article {article_id}...")
       
       # 1. Get recommendations
        visual_recs = self._get_visual_recs(article_id)
        print("Visual recs:", visual_recs)

        content_recs = self._get_content_recs(article_id)
        print("Content recs:", content_recs)

        collaborative_recs = self._get_collaborative_recs(user_id)
        print("Collaborative recs:", collaborative_recs)
        
        
       
       
       
       
       
        # 1. Get recommendations from all models
        visual_recs = self._get_visual_recs(article_id)        # Call visual recommender: returns list of similar articles based on image embeddings
        content_recs = self._get_content_recs(article_id)      # Call content-based recommender: returns list based on TF-IDF similarity of product descriptions
        collaborative_recs = self._get_collaborative_recs(user_id)  # Call collaborative filtering recommender: returns list of personalized articles based on user's past interactions

        # 2. Apply the weighted scoring strategy
        recommendation_scores = {}  # Initialize an empty dictionary to store each article ID and its total "score" from all models

        # Give points for each article recommended by the visual model
        for rec_id in visual_recs:
            # If rec_id is already in recommendation_scores, add 3 points; otherwise start at 0 and add 3
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 3

        # Give points for collaborative filtering recommendations
        for rec_id in collaborative_recs:
            # Add 2 points for each article recommended by CF model, cumulative with any previous score
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 2

        # Give points for content-based recommendations
        for rec_id in content_recs:
            # Add 1 point for each article recommended by content-based model, cumulative with any previous score
            recommendation_scores[rec_id] = recommendation_scores.get(rec_id, 0) + 1

        # 3. Sort and rank the results
        # recommendation_scores.items() gives a list of tuples: (article_id, total_score)
        # sorted() orders these tuples by the total_score (item[1]), from highest to lowest because reverse=True
        sorted_recs = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)

        # 4. Return the top N article IDs, excluding the input article itself
        # Unpack each tuple (rec_id, score), skip the article_id that was used as input, and take only the article IDs
        final_recs = [rec_id for rec_id, score in sorted_recs if rec_id != article_id]

        # Limit to top N articles and return as the final hybrid recommendation list
        return final_recs[:n]

        
       