# AI-Powered Fashion Recommender 👗🧥👟

An internship project to build a **hybrid ML-powered fashion recommender system** that combines collaborative filtering, content-based filtering, and computer vision to deliver personalized outfit suggestions.

---

## 📖 Overview
This project was developed during a 12-week internship.  
The goal was to design and implement a **hybrid recommender system** that learns from user behavior, product descriptions, and clothing images to recommend the most relevant fashion items.

---

## ✨ Features
- **Popularity-Based Recommender**: Simple baseline using the most purchased items.  
- **Content-Based Recommender**: Uses TF-IDF and cosine similarity on product descriptions.  
- **Collaborative Filtering**: Deep learning model with embeddings for users and articles.  
- **Visual Recommender**: CNN-based image feature extraction for style similarity.  
- **Hybrid Recommender**: Combines all strategies into one system with weighted scoring.  
- **Streamlit Web App**: Interactive demo for trying out recommendations in real time.  

---

## 🛠 Tech Stack
- **Python**: Core programming language  
- **Pandas / NumPy**: Data handling  
- **Scikit-learn**: TF-IDF, similarity measures  
- **TensorFlow / Keras**: Collaborative filtering and CNN models  
- **Streamlit**: Frontend demo app  
- **Joblib / Pickle**: Artifact saving and loading  

---

## 🚀 How to Run
Follow these steps to run the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ai-fashion-recommender.git
cd ai-fashion-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
