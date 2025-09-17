# -------------------- Imports --------------------
import streamlit as st                         # Streamlit → creates the interactive web app UI
import pandas as pd                            # Pandas → handles reading and manipulating datasets
from PIL import Image                          # PIL (Python Imaging Library) → loads and works with images
from io import BytesIO                         # BytesIO → allows reading raw image bytes in memory (no need to save files)
import zipfile                                 # zipfile → to read images directly from compressed images.zip
import os                                      # os → helps with path and file operations
from src.recommender import HybridRecommender  # Import your custom recommendation engine

# -------------------- Config / File paths --------------------
ARTICLES_CSV = "data/articles.csv"  # Path to the articles metadata CSV (product details)
ZIP_PATH = "data/images.zip"        # Path to the H&M dataset images (compressed zip file)
NUM_DISPLAY = 12                    # Number of recommendations to display
NUM_COLS = 4                        # Number of images to show per row in grid layout

# -------------------- Helper: load articles CSV --------------------
@st.cache_data  # Cache results so CSV is only read once (speeds up the app)
def load_articles(csv_path):
    """
    Load the articles.csv and normalize article_id as a zero-padded string column 'article_id_str'.
    This ensures IDs like '108775015' always match with 10-character IDs in the dataset.
    """
    df = pd.read_csv(csv_path)  # Load CSV into DataFrame

    # Ensure 'article_id' column exists (different Kaggle versions sometimes name it differently)
    if 'article_id' not in df.columns:
        for candidate in ['articleid', 'article id', 'id']:
            if candidate in df.columns:
                df['article_id'] = df[candidate]
                break
    if 'article_id' not in df.columns:
        raise ValueError("articles.csv must contain an article id column (e.g. 'article_id').")

    # Convert article_id to zero-padded string (10 digits, e.g. 0001087750)
    df['article_id_str'] = df['article_id'].astype(str).str.zfill(10)
    return df

# -------------------- Helper: build image index --------------------
@st.cache_data  # Cache index to avoid rebuilding it every run
def build_image_index(zip_path):
    """
    Build a dictionary mapping {article_id_str : [list of image paths inside zip]}.
    Example:
      "0001087750" → ["images/010/877/501.jpg"]
    """
    index = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():  # Loop over all files inside the zip
            if member.lower().endswith('.jpg'):  # Only keep image files
                base = os.path.splitext(os.path.basename(member))[0]  # Get filename without extension
                index.setdefault(base, []).append(member)  # Add mapping: article_id → image path
    return index

# -------------------- Helper: extract product name --------------------
def extract_product_name_from_row(row):
    """
    Pick the best product name column from the CSV row.
    Falls back to showing 'Article <id>' if no product name available.
    """
    candidates = ['prod_name', 'product_name', 'name', 'detail_desc',
                  'product_group_name', 'index_name']
    for col in candidates:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])  # Use the first non-empty product name
    return f"Article {row['article_id_str']}"

# -------------------- Cache/load recommender engine --------------------
@st.cache_resource
def load_engine():
    """
    Load the HybridRecommender model.
    Cached so it initializes only once when the app starts.
    """
    return HybridRecommender()

# -------------------- App UI --------------------
st.title("🛍 AI-Powered Fashion Recommender")  # Title of the app
st.write("This app combines three models into hybrid recommendations and shows images from the H&M dataset zip.")

st.header("Enter Your Details")  # Section header
# Dropdown to select customer ID
test_user = st.selectbox(
    "Choose a Customer ID:",
    ['0000423b00ade91418cceaf3b26c6af3dd342b51fd051eec9c12fb36984420fa', '000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318', '00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2', '0000f1c71aafe5963c3d195cf273f7bfd50bbf17761c9199e53dbb81641becd7', '00015c1a121e08bbd2552c15fbbb6e6b19d3bf8f7b6a3d60c6d7be26f06264d6']
)
# Input box for entering a liked article
test_article = st.text_input("Enter an Article ID you like:", 573086001)
# Button to trigger recommendation
run_button = st.button("Get Recommendations")

# -------------------- Preload data & resources --------------------
# Load articles.csv into DataFrame
try:
    articles_df = load_articles(ARTICLES_CSV)
except Exception as e:
    st.error(f"Could not load articles CSV: {e}")
    st.stop()  # Stop app execution if CSV fails

# Build index of images from images.zip
try:
    image_index = build_image_index(ZIP_PATH)
except Exception as e:
    st.error(f"Could not open images zip: {e}")
    st.stop()

# Debug section: expand to preview CSV columns and first few rows
with st.expander("CSV columns & sample (debug)"):
    st.write("CSV columns:", list(articles_df.columns))
    st.dataframe(articles_df.head(3))

# Load recommender engine (cached)
recommender_engine = load_engine()

# -------------------- Main: when button clicked --------------------
if run_button:
    st.write("🔎 Finding recommendations... Please wait!")

    try:
        # Get top N recommendations
        recommendations = recommender_engine.get_recommendations(test_user, test_article, n=NUM_DISPLAY)

        st.header("Here are your personalized recommendations:")

        if not recommendations:  # No results found
            st.warning("Sorry, no recommendations found.")
        else:
            cols = st.columns(NUM_COLS)  # Create 4-column layout for displaying results

            with zipfile.ZipFile(ZIP_PATH, 'r') as z:
                # Loop through recommended items
                for i, rec_id in enumerate(recommendations[:NUM_DISPLAY]):
                    rec_id_str = str(rec_id).zfill(10)  # Normalize ID (10-digit string)

                    # ----- Step 1: Look up product name -----
                    row_mask = articles_df['article_id_str'] == rec_id_str
                    product_name = None
                    image_internal_path = None
                    if row_mask.any():
                        row = articles_df.loc[row_mask].iloc[0]
                        product_name = extract_product_name_from_row(row)

                        # Try using CSV's image columns if available
                        for col in ['image_path', 'image', 'image_name']:
                            if col in articles_df.columns:
                                candidate = str(row[col]).strip()
                                possibilities = [candidate]
                                if not candidate.startswith("images/"):
                                    possibilities.append(f"images/{candidate}")  # Normalize path
                                # Check if candidate path exists inside zip
                                for p in possibilities:
                                    if p in z.namelist():
                                        image_internal_path = p
                                        break
                                if image_internal_path:
                                    break

                    # ----- Step 2: If no image path from CSV, use prebuilt index -----
                    if image_internal_path is None:
                        paths = image_index.get(rec_id_str) or image_index.get(rec_id_str.lstrip('0'))
                        if paths:
                            image_internal_path = paths[0]  # Take first matching image

                    # ----- Step 3: Prepare caption -----
                    caption = product_name if product_name else f"Article {rec_id_str}"

                    # ----- Step 4: Display image or fallback -----
                    with cols[i % NUM_COLS]:  # Place item in correct column
                        if image_internal_path:  # If image exists
                            try:
                                image_bytes = z.read(image_internal_path)  # Read image bytes from zip
                                image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Load image in memory
                                st.image(image, caption=caption, use_container_width=True)  # Show image with name
                            except Exception as e_img:
                                st.write(caption)
                                st.info(f"⚠ Could not load image for {rec_id_str}")
                        else:  # If no image found
                            st.write(caption)
                            st.info("⚠ Image not found in zip")

    except Exception as e:
        # If recommendation process fails → show error and traceback
        st.error(f"⚠ Error generating recommendations: {e}")
        st.exception(e)
