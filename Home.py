import streamlit as st

# page configuration

st.set_page_config(
    page_title="Rooms&Rentals",
    layout="wide"  
)

st.markdown(
        """
        <style>
            img {
                border-radius: 12px;
                width: 1500px; 
                height: 500px;
                object-fit: Cover;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Home pgae

title = '<p style="font-family:sans-serif; color:#FF4B4B; font-size: 180px; font-weight: 600; ">Airbnb</p>'
st.markdown(title, unsafe_allow_html=True)


st.image("image-berlin.jpg",
     #    use_column_width= True,
     #    width = 400, 
         caption="Photo Credit Florian Wehde via Unsplash")











