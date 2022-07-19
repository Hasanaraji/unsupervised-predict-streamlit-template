from matplotlib.image import FigureImage
import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import requests
import re

# This code is different for each deployed app.
CURRENT_THEME = "blue"
IS_DARK_THEME = True
EXPANDER_TEXT = """
    This is a custom theme. You can enable it by copying the following code
    to `.streamlit/config.toml`:
    ```python
    [theme]
    primaryColor = "#E694FF"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#C6CDD4"
    font = "sans-serif"
    ```
    """


# This code is the same for each deployed app.
st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/apple/271/artist-palette_1f3a8.png",
    width=100,
)

"""
# Exploratory Data Analysis!
Lets Visualise Our data 👀 . 
You can select the theme you are comfortable with
"""

with st.expander("How can I use this theme in my app?"):
    st.write(EXPANDER_TEXT)

""
""

# Draw some dummy content in main page and sidebar.
def draw_all(
    key,
    plot =False,
):
    st.write(
        """
        What is Exploratory data analysis
    
        ```
        Exploratory data analysis is an approach to analyzing data sets to summarize their
        main characteristics, often with visual methods. EDA is the critical process of 
        performing initial investigations on data to discover patterns, to spot anomalies,
        to test hypothesis and to check assumptions with the help of summary statistics 
        and graphical representations.
        A statistical model can be used or not, but primarily EDA is for seeing what the 
        data can tell us beyond the formal modelling or hypothesis testing task.
        
        It is a good practice to understand the data first and try to gather as many 
        insights from it. EDA is all about making sense of data in hand, before getting
        them dirty with it, which will be done below."
        
        ```
        PREPARE TO GET YOU MIND BLOWN "💥"
        """
    )
    @st.cache
    def load_data(nrows):
        movie = pd.read_csv('resources/data/movies.csv')
        ratings = pd.read_csv('resources/data/ratings.csv')
        ratings.drop(['timestamp'], axis=1,inplace=True)
        data = ratings.merge(movie, on = "movieId")
        data = data.head(nrows)
        return data
    
    data_load_state = st.text('Loading data...')
    Movie = load_data(1000)
    ratings = pd.DataFrame(Movie.groupby('title')['rating'].mean())
    ratings['num_of_ratings'] = pd.DataFrame(Movie.groupby('title')['rating'].count())

    st.subheader("Movie preview")
    if st.checkbox("Show preview of data", key=key):
        st.subheader("Preview of movies")
        st.dataframe(ratings)
    #st.write(center_info_data)
    

    
    st.subheader('Ratings Chart')
    st.write(
        " #### Iteractive visualization of 21 movies User Ratings Chart ordered alphabetically"
    )
    
    #Bar Chart
    st.bar_chart(ratings['num_of_ratings'])
    
    st.write(
        " #### Bar chart showing the most rated and the least rated movies"
    )
    
    
    Movies = load_data(200000)
    user_rated = pd.DataFrame(Movies['movieId'].value_counts().reset_index()) # Create a user dataframe using groupby function

    user_rated.rename(columns = {'index':'movieId','movieId':'voted'},inplace = True) # Rename the columns 
    train = Movies.copy()
    train = train.merge(user_rated, on ='movieId') # Combine the train dataset with the User_rated data

    # Filter the data 
    train = train[train['voted'] > 100] # Find the movies which have the us voted for more than 50 

    train = train.sort_values('rating',ascending=False) # Sort the values by the rating feature

    
    fig, ax  = plt.subplots(1,2,figsize=(20,10)) # Initialize a figure 

    sns.barplot(ax=ax[0], x='rating',y = 'title', data=train.head(10)) # Create a bar plot of the top 10 movies
    ax[0].set_title('The top best rated movies ') # Set title of the bar plot 

    sns.barplot(ax=ax[1],x = 'rating',y = 'title', data = train.tail(10)) # Create a bar plot of the top 10 worst rated movies
    ax[1].set_title('The top worst rated movies ') # Set title of the bar plot 

    fig.tight_layout() # Set layout of the bar subplots 
    plt.show() # show the plots 
    st.pyplot(fig)


    st.write(
        " #### Barchart chart showing the Ditribution of the ratings"
    )    
    
    with sns.axes_style('white'):
        g = sns.factorplot("rating", data=Movies, aspect=2.0,kind='count')
        g.set_ylabels("Total number of ratings")
        st.write(f'Average rating in dataset: {np.mean(Movies["rating"])}')    
        st.pyplot(g)
    
    
    def top_n_plot_by_ratings(df,column, n):
        fig = plt.figure(figsize=(14,7))
        data = df[str(column)].value_counts().head(n)
        ax = sns.barplot(x = data.index, y = data, order= data.index, palette='CMRmap', edgecolor="black")
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')
        plt.title(f'Top {n} {column.title()} by Number of Ratings', fontsize=14)
        plt.xlabel(column.title())
        plt.ylabel('Number of Ratings')
        plt.xticks(rotation=90)
        plt.show()
        st.pyplot(fig)

    st.write(
        " #### Barchart chart showing the Ditribution of the Movie Title by Ratings"
    )    

    top_n_plot_by_ratings(Movies, "title", 10)



    
    st.write(
        " #### Barchart chart showing the Ditribution of the  UserID by Ratings "
    )    

    def user_ratings_count(df, n):
        fig = plt.figure(figsize=(14,7))
        data = df['userId'].value_counts().head(n)
        ax = sns.barplot(x = data.index, y = data, order= data.index, palette='CMRmap', edgecolor="black")
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=11, ha='center', va='bottom')
        plt.title(f'Top {n} Users by Number of Ratings', fontsize=14)
        plt.xlabel('User ID')
        plt.ylabel('Number of Ratings')
        plt.show()
        st.pyplot(fig)

    user_ratings_count(Movies, 10)
    
    st.write(
        " #### Bar chart showing the Distribution of movie genres"
    )
    
    
    #Distribution of movie genres
    fig = plt.figure(figsize=(20,7))
    genrelist = Movies['genres'].apply(lambda genrelist_movie : str(genrelist_movie).split("|"))
    genres_count = {}

    for genrelist_movie in genrelist:
        for genre in genrelist_movie:
            if(genres_count.get(genre,False)):
                genres_count[genre]=genres_count[genre]+1
            else:
                genres_count[genre] = 1       
    plt.bar(genres_count.keys(), genres_count.values(), color= "m")
    plt.show()
    st.pyplot(fig)

    st.checkbox("Is this cool or what?", key=key)
    st.radio(
        "How many balloons would you rate this?",
        ["1 balloon 🎈", "2 balloons 🎈🎈", "3 balloons 🎈🎈🎈"],
        key=key,
    )
    if st.button("🤡 Click me if you had fun", key=key):
        st.write("We are glad you had a blast")

    st.slider(
        "From 1 to 10, how cool was your experience?", min_value=1, max_value=10, key=key
    )
    st.text_area("A little writing space for your feedback:)", key=key)

    with st.expander("Expand me!"):
        st.write("Hey there! Nothing to see here 👀")
    st.write("")
    
    st.write("This is the end. We hope to see you again!")


draw_all("main", plot=True)

with st.sidebar:
    draw_all("sidebar")
    
    
