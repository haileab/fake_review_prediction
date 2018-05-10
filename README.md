# Galvanize Capstone Project -  Fake Review Prediction

----------- Summary -----------
After 8 weeks of learning data science at Galvanize we were tasked with building completing an individual Capstone Project.

Everyone uses reviews to make an informed decision about where to eat, what hotel to stay at, or what product to buy. After reading about fake reviews found on Amazon I wanted to see if machine learning can be used to predict fake reviews.

The data set that I used consists of 300,000 labeled restaurant reviews from Yelp.

----------- Modeling -----------
The data contained limited amount of user and business information and this project required heavy feature engineering. I used cumulative sum to produce historical review information.This is information that Yelp maintains and likely uses for their algorithm but wasn’t provided in the data set.

NLP plays a key a role in the development of AI based analytics and the analysis of text. NLP training and tuning was a large part of this project. I was able to test several parameters and models in an attempt to improve the NLP results.

----------- Results -----------
A Random Forest model was able to correctly classify 44% of fake reviews as fake, while incorrectly classifying only 6% of genuine reviews as fake.

NLP model didn’t provide significant signal and wasn’t used in the final model.   

----------- Next Steps -----------
Determine a cost-benefit matrix to filtering genuine or fake reviews that takes into consideration all of the stakeholders for Yelp. This can be used to set a preferred threshold where you can balance out the needs of users, reviewers, and businesses on the platform.

Thank you!
