import pickle
import streamlit as st

# Function to load models and similarity based on the selected category


def load_models(selected_category):
    if selected_category == 'Actor':
        model_location = 'models/Based-Actor/movie_list.pkl'
        similarity_location = 'models/Based-Actor/similarity.pkl'
    elif selected_category == 'Director':
        model_location = 'models/Based-Director/movie_list.pkl'
        similarity_location = 'models/Based-Director/similarity.pkl'
    else:
        model_location = 'models/Based-Genre/movie_list.pkl'
        similarity_location = 'models/Based-Genre/similarity.pkl'

    # Load the models
    movies = pickle.load(open(model_location, 'rb'))
    similarity = pickle.load(open(similarity_location, 'rb'))

    return movies, similarity

# Define the movie recommendation function


def recommend(movie, movies, similarity):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    # Get top 5 recommendations (excluding the first one, which is the same movie)
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names


# Streamlit UI setup
st.header('Movie Recommender System')

# Define the categories for recommendation
recommend_category = ['Genre', 'Director', 'Actor']
selected_category = st.selectbox(
    "Select recommendation category", recommend_category)

# Load the appropriate models and similarity matrix based on the selected category
movies, similarity = load_models(selected_category)

# Get the list of movie titles for the dropdown
movie_list = movies['title'].values

# Movie selection by the user
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown", movie_list)

# Show recommendations when the button is pressed
if st.button('Show Recommendation'):
    recommended_movie_names = recommend(selected_movie, movies, similarity)

    # Display recommended movie names
    st.markdown("### Recommended Movies:")
    for movie in recommended_movie_names:
        st.markdown(f"- {movie}")
