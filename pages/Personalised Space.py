import streamlit as st
import pandas as pd
from Home import content_rec

# load data
listings_df = pd.read_csv('listings_cl.csv')



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

#----------------------------------------------------------------------------------------------------------------------
# Content based
#----------------------------------------------------------------------------------------------------------------------


#st.header("Personalize Your Stay Here!")


title_2 = '<p style="font-family:sans-serif; color:#FFA500; font-size: 42px; font-weight: 600; ">Personalize Your Stay Here!</p>'
st.markdown(title_2, unsafe_allow_html=True)


# formatting, create 3 columns
#col1, col2  = st.columns([1,1])

# header
# add image
#col2.image("image-2.jpg", width = 410, caption = "Photo Credit Patrick Reichboth via Unsplash. Location: Prenzlauer Berg, Berlin, Germany")


subtitle1 = '<p style="font-size: 20px;">Using the dropdown menus, sliders, and by explaining your preference get recommendations.</p>'
st.markdown(subtitle1, unsafe_allow_html = True)
#col1.markdown(subtitle1, unsafe_allow_html = True)


# GET USER INPUTS #
# ----------------------------------------------- #
neighborhood = st.selectbox('Location:',(listings_df['neighbourhood_cleansed'].unique()))

unique_accommodates = sorted(listings_df['accommodates'].unique())
accommodation = st.selectbox('Number of Guests:',(unique_accommodates))

# Create two input boxes for minimum and maximum prices side by side

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

    # display similar recommendation
    # divide page into 5 columns to display top 5
    columns = st.columns(5)

    # Display the listings in each column
    for i in range(5):
        with columns[i]:
            listing = similar_rentals.iloc[i]
            st.image(listing["picture_url"],
                     use_column_width=False,
                     width = 130,
                     caption='', clamp=False,
                     output_format='JPEG')
            st.write(f'**{listing["name_0"]}**')
            st.write(f'**<p style="text-align: left;"><font color="#FFA500">★{listing["review_scores_rating"]}({listing["number_of_reviews"]})</font></p>**', unsafe_allow_html=True)  # ##4B853C
            st.write(f'{listing["description_cleaned"][:120]}')
            st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FFA500;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)

