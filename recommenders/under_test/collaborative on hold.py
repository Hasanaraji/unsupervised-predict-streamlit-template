"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""
# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import random
from itertools import chain
import streamlit as st
import requests
import pickle

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
ratings.drop(['timestamp'], axis=1,inplace=True)

def load_movie_titles(path_to_movies):
    """Load movie titles from database records.

    Parameters
    ----------
    path_to_movies : str
        Relative or absolute path to movie database stored
        in .csv format.

    Returns
    -------
    list[str]
        Movie titles.

    """
    df = pd.read_csv(path_to_movies)
    df = df.dropna()
    df['title'] = df['title'].str[:-7]
    movie_list = df['title'].to_list()
    return movie_list


# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
def preprocessing(ratings):
    final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
    final_dataset.fillna(0,inplace=True)

    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

    #final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    return final_dataset, knn, csr_data

def iterFlatten(root):
    if isinstance(root, (list, tuple)):
        for element in root:
            for e in iterFlatten(element):
                yield e
    else:
        yield root

def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (st-/r)
        Titles of the top-n movie recommendations to the user.

    """
    final_dataset, knn, csr_data = preprocessing(ratings)
    movie_l = []
    for movie_name in movie_list:
        n_movies_to_reccomend = top_n
        movie_list = movies[movies['title'].str.contains(movie_name)]
        movie_l.append(movie_list)
    if len(movie_l):
        try:
            movie_idx1= movie_l[0].iloc[0]['movieId']
        except Exception:
            pass
        try:
            movie_idx2= movie_l[1].iloc[0]['movieId']
        except Exception:
            pass
        try:
            movie_idx3= movie_l[2].iloc[0]['movieId']
        except Exception:
            pass
        try:
            movie_idx1 = final_dataset[final_dataset['movieId'] == movie_idx1].index[0]
        except Exception:
            pass
        try:
            movie_idx2 = final_dataset[final_dataset['movieId'] == movie_idx2].index[0]
        except Exception:
            pass
        try:
            movie_idx3 = final_dataset[final_dataset['movieId'] == movie_idx3].index[0]
        except Exception:
            pass
        try:
            distances1 , indices1 = knn.kneighbors(csr_data[movie_idx1],n_neighbors=n_movies_to_reccomend+1)
        except Exception:
            pass
        try:
            distances2 , indices2 = knn.kneighbors(csr_data[movie_idx2],n_neighbors=n_movies_to_reccomend+1)
        except Exception:
            pass
        try:
            distances3 , indices3 = knn.kneighbors(csr_data[movie_idx3],n_neighbors=n_movies_to_reccomend+1)
        except Exception:
            pass
        try:
            rec_movie_indices1 = sorted(indices1.squeeze())
        except Exception:
            pass
        try:
            rec_movie_indices2 = sorted(indices2.squeeze())
        except Exception:
            pass
        try:
            rec_movie_indices3 = sorted(indices3.squeeze())
        except Exception:
            pass
        try:    #rec_movie_indices = rec_movie_indices1.append(rec_movie_indices2).append(rec_movie_indices3).sort_values(ascending = False)
            rec_movie_indices = list(iterFlatten(rec_movie_indices1))
        except Exception:
            pass
        try:
            rec_movie_indices = list(iterFlatten(rec_movie_indices2))
        except Exception:
            pass
        try:
            rec_movie_indices = list(iterFlatten(rec_movie_indices3))
        except Exception:
            pass
        recommend_frame = []
        random.shuffle(rec_movie_indices)
        for val in rec_movie_indices[:n_movies_to_reccomend]:
            movie_idx = final_dataset.iloc[val]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append(movies.iloc[idx]['title'].values[0])
        df = list(recommend_frame)
        return df
    else:
        return "No movies found. Please check your input"
 
 
movie_list = load_movie_titles('resources/data/movies.csv')
selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
            movie_list    )    


movies_dict=pickle.load(open('movie_link_list.pkl','rb'))
movies_df=pd.DataFrame(movies_dict)

def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=b6276f25a72128d4d93877a8b540e2c6&language=en-US'.format(movie_id))
    data=response.json()
    return "http://image.tmdb.org/t/p/w500/"+data['poster_path']
    
def recommend(selected_movie):
    movie_list = list(selected_movie.split("%%"))
    distances = collab_model(movie_list)
    recommended_movie_names = []
    recommended_posters=[]

    for i in distances:
        movie_list = list(i.split("%%"))
        name_list = []
        for i in movie_list:
            name = movies_df[movies_df['title'].str.contains(i)]
            name_list.append(name)
        movie_idx = name_list[0].iloc[0]['title']
        idx = movies_df[movies_df['title'] == movie_idx].index
        recommended_movie_names.append(movies_df.iloc[idx]['title'].values[0])
        recommended_posters.append(fetch_poster(movies_df.iloc[idx]['tmdbId'].values[0]))
    return recommended_movie_names, recommended_posters

names,posters=recommend(selected_movie)

col1,col2,col3,col4,col5=st.columns(5)
    
with col1:
    st.text(names[0])
    st.image(posters[0])
with col2:
    st.text(names[1])
    st.image(posters[1])
with col3:
    st.text(names[2])
    st.image(posters[2])
with col4:
    st.text(names[3])
    st.image(posters[3])
with col5:
    st.text(names[4])
    st.image(posters[4])
    
with col1:
    st.text(names[5])
    st.image(posters[5])
with col2:
    st.text(names[6])
    st.image(posters[6])
with col3:
    st.text(names[7])
    st.image(posters[7])
with col4:
    st.text(names[8])
    st.image(posters[8])
with col5:
    st.text(names[9])
    st.image(posters[9])
