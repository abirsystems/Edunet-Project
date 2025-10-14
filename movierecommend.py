import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests

# Fetch movie poster from TMDB
def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=6cb41288966ad746fb4f14a16e73912a".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500"+data['poster_path']

# Function to recommend movies
def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies = []
    recommmended_posters = []   
    for i in movies_list:
        movie_id = movies.iloc[i[0]].id
        # Fetch poster from api
        recommended_movies.append(movies.iloc[i[0]].title)
        recommmended_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommmended_posters

# Loading movies dataset
movies = pickle.load(open('movies.pkl', 'rb'))
url = "https://huggingface.co/datasets/ml-enthusiast123/edunet-assets/resolve/main/similarity.pkl"
similarity = pickle.loads(requests.get(url).content)

movies_names = movies['title'].values

# UI
st.title("Movie Recommendation System")
option = st.selectbox('Type or Select  a movie from the dropdown : ', movies_names)

# Button to get recommendations
if st.button('Show Recommendations'):
    recommended_movie_names,recommended_movie_posters = recommend(option)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])



