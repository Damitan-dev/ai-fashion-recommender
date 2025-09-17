Project Goal: To build a hybrid recommender system combining content, collaborative, and visual models.
Inputs: customer_id, article_id.
Outputs: A ranked list of recommended article_ids.
Proposed Steps:
Create a master "recommender" class or script.
Load all necessary pre-trained models and data (TF-IDF matrix, CF model, CNN feature extractor).
Implement a function that takes inputs and calls each of the three models.
Develop and implement a blending/ranking algorithm.
Create a simple interface to test the final system.
