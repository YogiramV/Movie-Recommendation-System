import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
import nltk
import ast
import numpy as np
import pandas as pd

movies = pd.read_csv(
    "/home/yogi/workspace/Movie-Recommendation-System/Datasets/tmdb_5000_movies.csv")
credits = pd.read_csv(
    "/home/yogi/workspace/Movie-Recommendation-System/Datasets/tmdb_5000_credits.csv")


movies = movies.merge(credits, on="title")
movies = movies[['movie_id', 'title', 'overview',
                'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)


# Convert To List
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])

    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# Extract the top 3 cast
def convert_cast(obj):
    count = 0
    L = []
    for i in ast.literal_eval(obj):
        if count != 1:
            L.append(i['name'])
            count += 1
        else:
            break

    return L


movies['cast'] = movies['cast'].apply(convert_cast)


# Extract the director
def convert_crew(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break

    return L


movies['crew'] = movies['crew'].apply(convert_crew)


# Convert string to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())


# Remove spaces
movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x])


# Merge columns

# Based on Genre, Overview
movies['tags'] = movies['cast'] + movies['crew'] + movies['overview'] + movies['genres'] + \
    movies['keywords']

# Based on director and actor
# movies['tags'] = movies['cast']


# Remove unwanted columns
new_df = movies[['movie_id', 'title', 'tags']]


# Convert string to list and make it lower case
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


# Stemming
ps = PorterStemmer()


def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)


new_df['tags'] = new_df['tags'].apply(stem)


cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


pickle.dump(new_df, open(
    '/home/yogi/workspace/Movie-Recommendation-System/models/Based-Genre/movie_list.pkl', 'wb'))
pickle.dump(similarity, open(
    '/home/yogi/workspace/Movie-Recommendation-System/models/Based-Genre/similarity.pkl', 'wb'))
