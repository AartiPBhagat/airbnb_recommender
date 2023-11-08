import streamlit as st
import pandas as pd
# Rest of your Streamlit app
pd.set_option('display.max_colwidth', None)
#-----------------------------------------------------------------------

# Set page configuration
st.set_page_config(
    page_title="Rooms&Rentals",
    layout="wide"  
)

st.markdown(
        """
        <style>
            img {
                border-radius: 12px;
                object-fit: Cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# layout with two columns
#col1, col2 = st.columns([1, 1])  # Adjust column widths as needed


# title in the first column (to the right of the image)
#with col1:
title = '<p style="font-family:sans-serif; color:#FF4B4B; font-size: 200px; font-weight: 600; ">Airbnb</p>'
st.markdown(title, unsafe_allow_html=True)

# image in the second column
#with col2:
st.image("image-berlin.jpg",
         use_column_width=True,
         width = 600, 
         caption="Photo Credit Florian Wehde via Unsplash")
    
    
from sklearn.preprocessing import MinMaxScaler

# load data
listings_df = pd.read_csv('listings_cl.csv')



#-----------------------------------------------------------------------
# popularity
#-----------------------------------------------------------------------

def get_top_pop(n=10):
    

    # Define the columns to keep
    keep_cols = ['id', 'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness', 'review_scores_location','number_of_reviews_l30d','property_type']

    # conditions
    is_available = listings_df['has_availability'] == 1
    is_superhost = listings_df['host_is_superhost'] == 1
    host_verified = listings_df['host_identity_verified'] ==1
    is_response = (~(listings_df['host_response_time']=='no response'))


    # Filter, select, and drop missing values in one step
    dfp = listings_df[(is_available)
                          & (host_verified)
                                & (is_response)
                                    & (is_superhost) ].dropna()

    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()

    # Define the columns to scale
    scale_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
                  'number_of_reviews','number_of_reviews_l30d']

    # Apply scaling to selected columns
    dfp[scale_cols] = scaler.fit_transform(dfp[scale_cols])

    # Define the weights for rating components
    weights = {
        'review_scores_rating': 0.50,
        'review_scores_cleanliness': 0.2,
        'review_scores_location': 0.2,
        'number_of_reviews': 0.04,
        'number_of_reviews_l30d':0.06
    }

    # Calculate the weighted sum of ratings
    dfp['final_rating'] = dfp[scale_cols].dot(pd.Series(weights))

    # Sort the DataFrame by the calculated 'final_rating' in descending order
    dfp = dfp.sort_values(by='final_rating', ascending=False)

    # Merge with the original DataFrame to get additional information
    top_listings = dfp[['id','final_rating']].merge(listings_df[['id', 'name_0', 'number_of_reviews', 'review_scores_rating', 'description_cleaned', 'listing_url','picture_url']], on='id', how='left')

    # Display the top 10 listings
    top_n_pop = top_listings.head(n)

    return top_n_pop

#-----------------------------------------------------------------------
# Content based
#-----------------------------------------------------------------------

import re
import string

# clean description column
listings_df['desc_proc'] = listings_df['description_cleaned'].apply(lambda text: text.translate(str.maketrans('','',string.punctuation)))

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the stopwords dataset (do this once)
nltk.download('stopwords')

# Load the stopwords for both English and German
stop_words_english = set(stopwords.words('english'))
stop_words_german = set(stopwords.words('german'))

# func for preprocess the description text
def preprocess_text(text):
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words_english and word not in stop_words_german]
    return ' '.join(tokens)

# Apply the preprocessing function to 'description_cleaned' column
listings_df['desc_proc'] = listings_df['desc_proc'].apply(preprocess_text)

# filter the original dataframe
listings_df = listings_df[(listings_df['has_availability'] == 1)&(listings_df['instant_bookable']==1)]

def content_rec(desired_neighbourhood,desired_number_person,low_price,high_price,input_text):
    # Step 1: Filter Listings
    filtered_listings = listings_df[(listings_df['accommodates'] == desired_number_person) &
                                   (listings_df['neighbourhood_cleansed'] == desired_neighbourhood) &
                                   (listings_df['price'] >= low_price) &
                                   (listings_df['price'] <= high_price)]
    input_text = "Apartment with balcony, cups, dishes, kitchen, coffee table, for 2 person, 3rd or above floor, sunny, wifi, tv"
    
    # Step 2: Calculate TF-IDF for Input Text and Filtered Listings
    tfidf_vectorizer = TfidfVectorizer(
        #max_df=1.0, min_df=1
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_listings['desc_proc'])
    input_tfidf = tfidf_vectorizer.transform([input_text])

    # Step 3: Calculate Cosine Similarity
    cosine_sim = cosine_similarity(input_tfidf, tfidf_matrix)

    # Step 4: Sort and Retrieve Top Similar Listings
    similar_listings_indices = cosine_sim[0].argsort()[::-1]  # Sort in descending order

    # columns
    display_columns = ['id','name_0','number_of_reviews','description_cleaned','accommodates','review_scores_rating','neighbourhood_cleansed','price','listing_url','picture_url']
    top_similar_listings = filtered_listings[display_columns].iloc[similar_listings_indices].head(10)

    return top_similar_listings

#-----------------------------------------------------------------------
#### style ####
#-----------------------------------------------------------------------











