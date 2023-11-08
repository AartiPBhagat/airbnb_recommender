import streamlit as st
import pandas as pd
# Rest of your Streamlit app
pd.set_option('display.max_colwidth', None)
from Home import get_top_pop

######################################################################################################################
#--------------------------------------------------------------------------------------------------------------------
# RECOMMENDER APP
#--------------------------------------------------------------------------------------------------------------------
######################################################################################################################



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
# popularity
#----------------------------------------------------------------------------------------------------------------------

#st.header(f"Top-rated Rentals") #<font color='#4B853C'>

title_1 = '<p style="font-family:sans-serif; color:#FFA500; font-size: 42px; font-weight: 600; ">Top-rated Rentals</p>'
st.markdown(title_1, unsafe_allow_html=True)

st.caption("These rentals received high rating for location, cleanliness, and more.")


# display top 10 listings
top_10_rentals = get_top_pop(10)
st.dataframe(top_10_rentals, hide_index=True)

    
# divide page into 5 columns to display top 5
columns = st.columns(5)



# Display the listings in each column
for i in range(5):
    with columns[i]:
        listing = top_10_rentals.iloc[i]
        image_width = 140
        st.image(listing["picture_url"],
                     use_column_width=False,
                     width = image_width,
                     caption='', clamp=False,
                     output_format='JPEG')
        st.write(f'**{listing["name_0"]}**')
        st.write(f'**<p style="text-align: left;"><font color="#FF4B4B">★{listing["review_scores_rating"]}({listing["number_of_reviews"]})</font></p>**', unsafe_allow_html=True)  # ##4B853C # #FFA500
        st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FFA500;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)

        
##############################



#############################


####------------ HOuse---------

#title_1 = '<p style="font-family:sans-serif; color:#FFA500; font-size: 42px; font-weight: 600; ">House</p>'
#st.markdown(title_1, unsafe_allow_html=True)


# display top 10 listings

#top_10_rentals = top_10_house_listings

#st.dataframe(top_10_rentals, hide_index=True)



