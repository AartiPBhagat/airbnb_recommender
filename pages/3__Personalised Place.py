import streamlit as st
import pandas as pd


# load data
listings_df = pd.read_csv('listings_rec.csv')

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
listings_df = listings_df[(listings_df['has_availability'] == 1)]

def content_rec(desired_neighbourhood,desired_number_person,low_price,high_price,input_text):
    
    # Step 1: Filter Listings
    filtered_listings = listings_df[(listings_df['accommodates'] == desired_number_person) &
                                   (listings_df['neighbourhood_cleansed'] == desired_neighbourhood) &
                                   (listings_df['price'] >= low_price) &
                                   (listings_df['price'] <= high_price)]

    # Step 2: take user input and clean it
    def clean_text(input_text):

        # Remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        cleaned_text = input_text.translate(translator)

        # Convert to lowercase and strip leading/trailing spaces
        cleaned_text = cleaned_text.lower().strip()

        return cleaned_text

    input_text = 'Apartment with bal%cony, (cups), dishesß, kitchen, öl, coffee table,for 2 person, 3rd or above floor, sunny, wifi, tv'
    cleaned_text = clean_text(input_text)

    # Step 3: Calculate TF-IDF for Input Text and Filtered Listings
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word') #, max_df=0.95, min_df=0
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_listings['desc_proc'])
    input_tfidf = tfidf_vectorizer.transform([cleaned_text])

    # Step 4: Calculate Cosine Similarity
    cosine_sim = cosine_similarity(input_tfidf, tfidf_matrix,)

    # Create a DataFrame with the cosine similarity scores
    cosine_sim_df = pd.DataFrame(cosine_sim[0], index=filtered_listings.id)

    # Step 5: Sort and Retrieve Top Similar Listings
    cosine_sim_df = cosine_sim_df.sort_values(by=0,ascending=False).reset_index()

    # columns
    display_columns =['id','name_0','number_of_reviews','description_cleaned','accommodates','review_scores_rating',
                      'neighbourhood_cleansed','price','listing_url','picture_url']

    #    top_similar_listings = filtered_listings[display_columns].iloc[similar_listings_indices].head(10)
    top_similar_listings = cosine_sim_df.merge(filtered_listings[display_columns],how='left')


    return top_similar_listings

####################################################################################
#-----------------------------------------------------------------------------------
# RECOMMENDER APP
#-----------------------------------------------------------------------------------
#####################################################################################



st.markdown(
        """
        <style>
            img {
                border-radius: 12px;
                width: 200px; 
                height: 150px;
                object-fit: Cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

#--------------------------------------------------------------------------------------
# Content based
#--------------------------------------------------------------------------------------


#st.header("Personalize Your Stay Here!")


title_2 = '<p style="font-family:sans-serif; color:#FF4B4B; font-size: 42px; font-weight: 600; ">Personalize Your Stay Here!</p>'
st.markdown(title_2, unsafe_allow_html=True)



subtitle1 = '<p style="font-size: 20px;">Using the dropdown menus, sliders, and by explaining your preference get recommendations.</p>'
st.markdown(subtitle1, unsafe_allow_html = True)


# GET USER INPUTS #

unique_neighborhood = sorted(listings_df['neighbourhood_cleansed'].unique())
unique_accommodates = sorted(listings_df['accommodates'].unique())

neighborhood = st.selectbox('Location:',(unique_neighborhood))
accommodation = st.slider('Number of Guests:', listings_df['accommodates'].min(), listings_df['accommodates'].max(), value=1)
col1, col2 = st.columns(2)
with col1:
    min_price = st.number_input("Minimum Price", min_value=0, max_value=800, value=100)
with col2:
    max_price = st.number_input("Maximum Price", min_value=0, max_value=800, value=200)
user_input = st.text_area("Describe your requirements",max_chars=200)



if st.button("Show me recommendations!"):
    
    # call fun to get similar listings
    similar_rentals = content_rec(desired_neighbourhood = neighborhood ,
                                  desired_number_person = accommodation ,
                                  low_price = min_price,
                                  high_price = max_price,
                                  input_text = user_input)

    if len(similar_rentals) ==0:
        st.write('Soory We can not find a good place for you.')
    elif len(similar_rentals) >= 5:
        n_cols = 5
    else :
        n_cols = len(similar_rentals)

    # divide page into 5 columns to display top 5
    columns = st.columns(n_cols)

    # Display the listings in each column
    for i in range(n_cols):
        with columns[i]:
            listing = similar_rentals.iloc[i]
            st.image(listing["picture_url"],
                     use_column_width=False,
                     width = 130,
                     caption='', clamp=False,
                     output_format='JPEG')
            st.write(f'**{listing["name_0"]}**')
            st.write(f'**<p style="text-align: left;"><font color="#FF4B4B">★{listing["review_scores_rating"]}({listing["number_of_reviews"]})</font></p>**', unsafe_allow_html=True)  # ##4B853C
            st.write(f'{listing["description_cleaned"][:120]}')
            st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FF4B4B;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)
        
