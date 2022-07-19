# -*- coding: utf-8 -*-
import pickle
import streamlit as st
import re
import pandas as pd
import requests



def Poster(title):
    if title:
        try:
            url = f"http://www.omdbapi.com/?t={title}&apikey=2818afea"
            re = requests.get(url)
            re = re.json()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(re["Poster"])
            with col2:
                st.subheader(re['Title'])
                st.caption(f"Genre: {re['Genre']} Year: {re['Year']} ")
                st.write(re['Plot'])
                st.text(F"Rating: {re['imdbRating']}")
                st.progress(float(re['imdbRating']))
        except:   
            st.error('') 

def clean(title):
    title = re.sub("[^a-zA-Z]", " ", title)
    return title

def recommend(title, movies, similarity):
    idx = indices.loc[title]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names

def data(df):
    movies = pickle.load(open(df,'rb'))
    return movies
def sim(sim):    
    similarity = pickle.load(open(sim,'rb'))
    return similarity

page_bg_img = '''
    <style>

          .stApp {
    background-image: url("https://imgur.com/USx83s6.png");
    background-size: cover;
    }
    </style>
    '''

st.markdown(page_bg_img, unsafe_allow_html=True)
    

st.markdown('# Movie Recommendation System')

movie_range = st.radio("Select the Name range of your movie",
                        ('0-B', 'B-D', 'D-H', 'H-L', 'L-N',
                        'N-S', 'S-T', 'T-#'))

if movie_range == '0-B':
    movies =data("movie0.pkl")
    similarity = sim("sim0.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'B-D':
    movies =data("movie1.pkl")
    similarity = sim("sim1.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'D-H':
    movies =data("movie2.pkl")
    similarity = sim("sim2.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'H-L':
    movies =data("movie3.pkl")
    similarity = sim("sim3.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'L-N':
    movies =data("movie4.pkl")
    similarity = sim("sim4.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'N-S':
    movies =data("movie5.pkl")
    similarity = sim("sim5.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'S-T':
    movies =data("movie6.pkl")
    similarity = sim("sim6.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 

if movie_range == 'T-#':
    movies =data("movie7.pkl")
    similarity = sim("sim7.pkl")
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    movie_list = movies['title']
    selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
        movie_list    )
    recomm = []
    if st.button('Show Recommendation'):
        st.write("#### You have chosen")
        Poster(selected_movie)
        st.write("#### We recommended you also see this movie(s)")
        recommended_movie_names = recommend(selected_movie, movies, similarity)
        for i in recommended_movie_names:
            Poster(i)
            recomm.append(i) 
