# -*- coding: utf-8 -*-
import pickle
import streamlit as st
import re
import pandas as pd
import requests
from streamlit_option_menu import option_menu

st.set_page_config(page_icon="https://imgur.com/ABwifID.png")

movies_dict=pickle.load(open('movie_link_list.pkl','rb'))
all_movies=pd.DataFrame(movies_dict)



def fetch_poster(movie_id):
    """Fetches the Image address link of a given Movie
    Parameters
    ----------
    movie_id (float)
        tmdbId of a particular movie.
    
    Returns
    -------
     url address path(str)   
    """
    response=requests.get('https://api.themoviedb.org/3/movie/{}?api_key=b6276f25a72128d4d93877a8b540e2c6&language=en-US'.format(movie_id))
    data=response.json()
    return "http://image.tmdb.org/t/p/w500/"+data['poster_path']

#---------------------------------------------------------------------------
# Defining a function to find Top Rated Movies by Popularity and User Rating
#---------------------------------------------------------------------------
def recommend_0(col):
    """Finds Popular movies by Popularity and User Rating and 
    attaches it's name and poster.

    Parameters
    ----------
    col : Datafame column (float)
        Favorite movies chosen by the app user.

    Returns
    -------
    recommended_movies : list (str)
        Titles of popular movies to the user.
    recommended_posters : list (str)
        URL addresses of the poster image
    """
    #sorting Dataframe values by selected column
    movies_list=all_movies.sort_values(by=col, ascending=False, inplace=False)
    movies_list=movies_list.iloc[0:10] #Taking only the top 10 Movies

    recommended_movies=[]
    recommended_posters=[]

    for i in range(len(movies_list)):
        id=movies_list.iloc[i,4] #Creating an Id for poster address link
        recommended_movies.append(all_movies[all_movies['tmdbId']==id].iloc[0,1])
        recommended_posters.append(fetch_poster(id))
    
    return recommended_movies,recommended_posters




def Poster(title):
    """Takes the Name or title of a movie and displays the poster, title, movie overview and mean rating
    with a scale of 0 to 10 of the selected movie  
    Prameters
    ---------
    
    """
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

# Function for removing
def clean(title):
    title = re.sub("[^a-zA-Z]", " ", title)
    return title



def recommend(title, movies, similarity):
    idx = indices.loc[title]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_posters=[]

    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movies.iloc[i[0]].tmdbId))
    return recommended_movie_names, recommended_posters

def data(df):
    movies = pickle.load(open(df,'rb'))
    return movies
def sim(sim):    
    similarity = pickle.load(open(sim,'rb'))
    return similarity

with st.sidebar:
    page = option_menu(
        menu_title = None,
        options = ["Home","Search"],
        icons = ["house","search"],
    )

page_bg_img = '''
    <style>

          .stApp {
    background-image: url("https://imgur.com/USx83s6.png");
    background-size: cover;
    }
    </style>
    '''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)

st.markdown('<h1 class="font">SkyWalker Recommendation System</h1>', unsafe_allow_html=True)


#--------------------------------------------------------
# 2. HOME PAGE
#--------------------------------------------------------

if page == "Home":

    st.write(" # Most Popular Movies")
    st.sidebar.image("https://media.giphy.com/media/3oeSAzp6JXsjSZdPZ6/giphy.gif")
    st.sidebar.write("""Not unless you can alter time, speed up the harvest or teleport me off this rock..--
         Luke Skywalker""")

    # Displays nme and image of popular movies
    names,posters=recommend_0('Popularity')

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
    
    #Displays name and video of current movies
    st.write(" # Trailers")
    st.text("")
    st.write(" #### Starwars")
    st.video("resources/videos/Obi-Wan.mp4")  
    st.text("")
    st.write(" #### She Hulk")
    st.video("resources/videos/She-Hulk.mp4")
            



#--------------------------------------------------------
# 3. SEARCH PAGE 
#--------------------------------------------------------

if page == "Search":


    st.markdown(""" <style> .pont {
    -size:18px ; font-family: 'Cooper Black'; color: #0000FF;} 
    </style> """, unsafe_allow_html=True)

    st.sidebar.write("""Attachment is forbidden. Possession is forbidden. 
         Compassion, which I would define as unconditional love, is essential to a Jedi's life. 
         So you might say, that we are encouraged to love.--
         Anakin Skywalker, Attack of the Clones, 2002.""")

    movie_range = st.sidebar.radio("SELECT FROM THE LIST",
                        ('0-B', 'B-E', 'E-H', 'H-L', 'L-P',
                        'P-S', 'S-U', 'U-#'))



    # Creation of movie ranges with chunked dataset.
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'B-E':
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'E-H':
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
            names, posters = recommend(selected_movie, movies, similarity)
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'L-P':
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'P-S':
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'S-U':
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
            names, posters = recommend(selected_movie, movies, similarity)
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

    if movie_range == 'U-#':
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
            names, posters = recommend(selected_movie, movies, similarity)
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
