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

# Importing data
movies = pd.read_csv('resources/data/movies.csv', sep = ',')
ratings = pd.read_csv('resources/data/ratings.csv')
movies.dropna(inplace=True)
ratings.drop(['timestamp'], axis=1,inplace=True)

# We make use of an KNearest-Neighbors to train the data set given.
def preprocessing(ratings):
    """Takes the ratings dataframe Preprocesses it and fits it using knn
    Parameters
    ----------
    ratings: pandas DataFrame
            The ratings dataframe containg userId, rating, and movieId
    Returns
    -------
    final_dataset: dataframe
            The final preprocessed dataset
    knn:  model
            The trained model using  K-nearest-neighbors (knn)
    csr_data: matrix array
            The sparse matrix of the final_dataset 
    
    """
    # Converting the datasetto a pivot table
    final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
    final_dataset.fillna(0,inplace=True)

    no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
    no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

    #final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
    #Taking the Sparse matrix of the dataset
    csr_data = csr_matrix(final_dataset.values)
    final_dataset.reset_index(inplace=True)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(csr_data)
    return final_dataset, knn, csr_data

def iterFlatten(root):
    """Takes a list of list or tuple and flattens it into a single list
    Parameters
    ----------
    root: list(str)
    
    Returns
    -------
    list(str) 
    """
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
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    final_dataset, knn, csr_data = preprocessing(ratings)
    #Creating a list of movies that cotains specified movie name
    movie_l = []
    for movie_name in movie_list:
        n_movies_to_reccomend = top_n
        movie_list = movies[movies['title'].str.contains(movie_name)]
        movie_l.append(movie_list)
    if len(movie_l):
        # Getting the index of the movie that matches the title
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
            #Getting the distances and nearest neighbors
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
        try:    
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
        #Shuffle the result so they are randomly printed out
        try:
            top_indexes = np.setdiff1d(rec_movie_indices,[movie_idx1,movie_idx2,movie_idx3])
        except:
            top_indexes = rec_movie_indices
            
        random.shuffle(top_indexes)
        
        for val in top_indexes[:n_movies_to_reccomend]:
            movie_idx = final_dataset.iloc[val]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append(movies.iloc[idx]['title'].values[0])
        df = recommend_frame
        return df
    else:
        return "No movies found. Please check your input"