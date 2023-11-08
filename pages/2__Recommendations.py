import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#from Home import get_top_pop

#----------------------------------------------------------------------------------------------------------------------
# popularity 
#----------------------------------------------------------------------------------------------------------------------

def get_top_pop(n=10):
    
    listings_df = pd.read_csv('listings_cl.csv')

    # Define the columns to keep
    keep_cols = ['id', 'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness', 
                 'review_scores_location','number_of_reviews_l30d','property_type']

    # conditions
    is_available = listings_df['has_availability'] == 1
    is_superhost = listings_df['host_is_superhost'] == 1
    host_verified = listings_df['host_identity_verified'] ==1
    is_response = (~(listings_df['host_response_time']=='no response'))


    # Filter, select, and drop missing values in one step
    dfp = listings_df[(is_available)
                          & (host_verified)
                                & (is_response)
                                    & (is_superhost)
                     ].dropna(subset=['review_scores_rating'])

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
        'review_scores_cleanliness': 0.22,
        'review_scores_location': 0.2,
        'number_of_reviews': 0.07,
        'number_of_reviews_l30d':0.01,
    }

    # Calculate the weighted sum of ratings
    dfp['final_rating'] = dfp[scale_cols].dot(pd.Series(weights))

    # Sort the DataFrame by the calculated 'final_rating' in descending order
    dfp = dfp.sort_values(by='final_rating', ascending=False)


    # Merge with the original DataFrame to get additional information
    top_listings = dfp[['id','final_rating','property_type']].merge(listings_df[['id', 'name_0', 'number_of_reviews', 'review_scores_rating', 'description_cleaned', 'listing_url','picture_url']], on='id', how='left')


    # Display the top 10 listings
    top_n_pop = top_listings.head(n)
    
    return top_n_pop





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

#---------------------------------------------Top --------------------------------------------------------------------


title_1 = '<p style="font-family:sans-serif ; font-size: 42px; font-weight: 600; ">Top-rated Rentals</p>'  #color:#FF4B4B
st.markdown(title_1, unsafe_allow_html=True)

#st.caption("These rentals received high rating for location, cleanliness, and more.")


# display top 10 listings
top_10_rentals = get_top_pop(10)


    
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
        st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FF4B4B;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)
        

        
##############################
#----------------------------------------------------------------------------------------------------------------------
# popularity style wise
#---------------------------------------------------------------------------------------------------------------------

#############################


listings_df = pd.read_csv('listings_cl.csv')

# Define the columns to keep
keep_cols = ['id', 'number_of_reviews', 'review_scores_rating', 'review_scores_cleanliness', 
             'review_scores_location','number_of_reviews_l30d','property_type']

# conditions
is_available = listings_df['has_availability'] == 1
#is_superhost = listings_df['host_is_superhost'] == 1
host_verified = listings_df['host_identity_verified'] ==1
#is_response = (~(listings_df['host_response_time']=='no response'))


# Filter, select, and drop missing values in one step
dfp = listings_df[(is_available)
                      & (host_verified)
                       #     & (is_response)
                        #        & (is_superhost)
                 ].dropna(subset=['review_scores_rating'])

# Create a MinMaxScaler instance
scaler = MinMaxScaler()

# Define the columns to scale
scale_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_location',
              'number_of_reviews']

# Apply scaling to selected columns
dfp[scale_cols] = scaler.fit_transform(dfp[scale_cols])

# Define the weights for rating components
weights = {
    'review_scores_rating': 0.50,
        'review_scores_cleanliness': 0.22,
        'review_scores_location': 0.2,
        'number_of_reviews': 0.08,
       
}

# Calculate the weighted sum of ratings
dfp['final_rating'] = dfp[scale_cols].dot(pd.Series(weights))

# Sort the DataFrame by the calculated 'final_rating' in descending order
dfp = dfp.sort_values(by='final_rating', ascending=False)


# Merge with the original DataFrame to get additional information
top_listings = dfp[['id','final_rating','property_type']].merge(listings_df[['id', 'name_0', 'number_of_reviews', 'review_scores_rating', 'description_cleaned', 'listing_url','picture_url']], on='id', how='left')



# Separate listings into three categories: House, Villas-Bungalows, and Boats 
house_listings = top_listings[top_listings['property_type'].isin(['Entire home','Entire vacation home','Entire guesthouse','Entire guest suite', 'Entire cottage' ])]
villas_bunglows_listings = top_listings[top_listings['property_type'].isin(['Entire villa','Entire bungalow','Entire chalet','Private room in villa','Private room in bungalow'])]
boats_listings = top_listings[top_listings['property_type'].isin(['Entire boat','Houseboat', 'Boat', 'Shared room in boat', 'Private room in boat', 'Private room in houseboat'])]




###########################################################################
#----- styles
###########################################################################




title_2_1 = '<p style="font-family:sans-serif; ; font-size: 42px; font-weight: 600; ">Rentals for Different Styles</p>'
st.markdown(title_2_1, unsafe_allow_html=True)



tab1, tab2, tab3 = st.tabs(["Houses", "Villas & Bungalows", "Boats"])
#---------------------------------------------House --------------------------------------------------------------------


with tab1:
    
    title_2_1 = '<p style="font-family:sans-serif; font-size: 32px; font-weight: 600; ">Houses</p>'
    st.markdown(title_2_1, unsafe_allow_html=True)


    # display top 10 listings houses
    top_10_houses_listings = house_listings.head(10)
    top_10_rentals =  top_10_houses_listings

    #st.dataframe(top_10_rentals, hide_index=True)



    # Display the listings in each column

    if len(top_10_rentals) >= 5:
        n_cols = 5
    else :
        n_cols = len(top_10_rentals)

    # divide page into 5 columns to display top 5
    columns = st.columns(n_cols)


    for i in range(n_cols):
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
            st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FF4B4B;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)



#--------------------------------------------- Villas & Bunglows --------------------------------------------------------------------

with tab2:

    title_2_1 = '<p style="font-family:sans-serif; font-size: 32px; font-weight: 600; ">Villas & Bunglows</p>'
    st.markdown(title_2_1, unsafe_allow_html=True)


    # display top 10 listings houses
    top_10_villas_bungalows_listings = villas_bunglows_listings.head(10)
    top_10_rentals =  top_10_villas_bungalows_listings

    #st.dataframe(top_10_rentals, hide_index=True)

    # Display the listings in each column

    if len(top_10_rentals) >= 5:
        n_cols = 5
    else :
        n_cols = len(top_10_rentals)

    # divide page into 5 columns to display top 5
    columns = st.columns(n_cols)




    # Display the listings in each column
    for i in range(n_cols):
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
            st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FF4B4B;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)
        
        
        
#--------------------------------------------- Boats --------------------------------------------------------------------

with tab3:

    title_2_1 = '<p style="font-family:sans-serif; font-size: 32px; font-weight: 600; ">Boats</p>'
    st.markdown(title_2_1, unsafe_allow_html=True)


    # display top 10 listings houses
    top_10_boats_listings = boats_listings.head(10)
    top_10_rentals =  top_10_boats_listings



    #st.dataframe(top_10_rentals, hide_index=True)

    # Display the listings in each column

    if len(top_10_rentals) >= 5:
        n_cols = 5
    else :
        n_cols = len(top_10_rentals)

    # divide page into 5 columns to display top 5
    columns = st.columns(n_cols)



    # Display the listings in each column
    for i in range(n_cols):
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
            st.markdown('<p style="font-size: 15px; color:#FFA500;"><a style="color:#FF4B4B;" href="{0}">More on Airbnb</a></p>'.format(listing["listing_url"]), unsafe_allow_html=True)
        
