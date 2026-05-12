import streamlit as st
import pickle
import pandas as pd

# Load data
movies = pickle.load(open('movies.pkl', 'rb'))
model = pickle.load(open("svd_model.pkl", 'rb'))

st.title("Hybrid Movie Recommendation System")

movie_name = st.selectbox("Select Movie", movies['title'].values)

user_id = st.number_input("Enter User ID", min_value=1, value=1)

if st.button("Recommend"):

    movie_index = movies[movies['title'] == movie_name].index[0]

    # SIMPLE hybrid logic: use item-based neighbors already stored OR fallback to SVD

    recommendations = []

    for i in range(len(movies)):

        movie_id = movies.iloc[i]['movieId']
        title = movies.iloc[i]['title']

        predicted_rating = model.predict(user_id, movie_id).est

        recommendations.append((title, predicted_rating))

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    st.subheader("Recommended Movies")

    for movie in recommendations[:10]:
        st.write(movie[0])
