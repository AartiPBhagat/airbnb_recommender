# Airbnb_recommender system

This airbnb recommender suggest popular airbnbs based on ratings and reviews. Apart from that one more layer or filtering is incorporated by providing style wise poular airbnbs.
It also provides personalized recommendation using NLP techniques.

1. Pooularity based Recommendation : A recommender that goes beyond mere ratings and reviews. This system considers a range of features, including host status, availability, and different kind of review scores i.e. cleanliness and location. To understand it's importance for deciding listing's popularity, regression analysis was performed.
2. Style wise Recommendations : Cosidering similar popular recommendations, one layer is added for three different categories of rentals : Houses, Villas & Bunglows, Boats(special ones).
3. Personalized Recommedations : Goal for this part is to create more personalised recommendations of Airbnb that matches a specific request. From user’s input, listings are filtered by price, capacity, location. Then, Fourth input from users is a text input specifying their unique requirements. Through the magic of tf-idf vectorization and cosine similarity, a personalised list of listings are suggested that whispered to the unique dreams of each traveller.
