# === BACKEND & DATA PREP ===
import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# Load CSV
def load_data(csv_file="imdb_movies_2024.csv"):
    df = pd.read_csv(csv_file)
    df.dropna(subset=["Title", "Description"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Clean Text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Build TF-IDF
def build_tfidf_matrix(df):
    df["cleaned_storyline"] = df["Description"].apply(clean_text)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["cleaned_storyline"])
    return tfidf, tfidf_matrix

# Recommend
def recommend_movies(input_storyline, df, tfidf, tfidf_matrix, top_n=5):
    cleaned_input = clean_text(input_storyline)
    input_vector = tfidf.transform([cleaned_input])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    results = df.iloc[top_indices][["Title", "Description"]].copy()
    results["Similarity Score"] = similarity_scores[top_indices]
    return results

# === STREAMLIT UI ===
@st.cache_data
def load_and_prepare_data():
    df = load_data("imdb_movies_2024.csv")
    tfidf, tfidf_matrix = build_tfidf_matrix(df)
    return df, tfidf, tfidf_matrix

def main():
    st.set_page_config(page_title="IMDb Movie Recommender", layout="centered")

    # === Custom Styling ===
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .movie-box {
            background-color: #ffffff;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 6px solid #FF4B4B;
            border-radius: 10px;
            box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
        }
        .movie-title {
            font-size: 22px;
            font-weight: bold;
            color: purple; /* 👈 Title above description */
            margin-bottom: 10px;
        }
        .similarity {
            font-size: 14px;
            color: #888;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # === Title & Instructions ===
    st.title("🎬 IMDb Movie Recommendation System (2024)")
    st.markdown(
        """
        Welcome to the **IMDb Movie Recommender**!  
        Enter a movie plot or storyline to discover similar films using advanced **TF-IDF** and **Cosine Similarity** techniques.

        ---
        """
    )

    # === Load Data ===
    df, tfidf, tfidf_matrix = load_and_prepare_data()

    # === User Input ===
    user_input = st.text_area("📖 Enter a movie storyline or plot:")

    # === Button & Results ===
    if st.button("🎯 Recommend Movies"):
        if user_input.strip() == "":
            st.warning("🚫 Please enter a storyline to get recommendations.")
        else:
            recommendations = recommend_movies(user_input, df, tfidf, tfidf_matrix, top_n=5)
            st.subheader("📽️ Top 5 Recommended Movies")

            for _, row in recommendations.iterrows():
                st.markdown(f"""
                    <div class='movie-box'>
                        <div class='movie-title'>🎬 {row['Title']}</div>
                        <div class='similarity'>Similarity Score: {row['Similarity Score']:.2f}</div>
                        <hr style='border: none; border-top: 1px solid #eee; margin: 10px 0;'/>
                        <p style='color: #333; line-height: 1.6;'>{row['Description']}</p>
                    </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
