import streamlit as st
import pickle
import pandas as pd

movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine.pkl', 'rb'))
model = pickle.load(open("svd_model.pkl", 'rb'))

st.title("Hybrid Movie Recommendation System")

movie_name = st.selectbox(
    "Select Movie",
    movies['title'].values
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    value=1
)

if st.button("Recommend"):

    movie_index = movies[movies['title'] == movie_name].index[0]

    similarity_scores = list(enumerate(cosine_sim[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x:x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:21]

    recommendations = []

    for movie in similarity_scores:

        index = movie[0]

        movie_id = movies.iloc[index]['movieId']

        title = movies.iloc[index]['title']

        predicted_rating = model.predict(user_id, movie_id).est

        recommendations.append((title, predicted_rating))

    recommendations = sorted(
        recommendations,
        key=lambda x:x[1],
        reverse=True
    )

    st.subheader("Recommended Movies")

    for movie in recommendations[:10]:
        st.write(movie[0])