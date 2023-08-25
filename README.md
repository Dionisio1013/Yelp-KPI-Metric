# Yelp KPI Web Application & Sentiment Analysis

If the website is down, hit the snooze button and wait a few minutes!

Here is a link to my [web application!](https://dionisio1013-hope-test-bdpevq.streamlit.app/)
This project is part of the [Yelp Dataset](https://www.yelp.com/dataset).

Here is a display of the website

<img width="1440" alt="image" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/44d5722a-8eb2-4640-ad1f-e349bca3604c">

<img width="1440" alt="image" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/78fd04b4-fbeb-4987-a7d1-90f46476d2c7">

<img width="1440" alt="image" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/d06c82e4-6b9d-4910-9b7e-4fe5df5446c6">

<img width="1440" alt="Screenshot 2023-08-21 at 3 39 57 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/367f29b1-dd5c-410b-8162-50242abd3cd0">

<img width="1440" alt="Screenshot 2023-08-21 at 3 38 48 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/ad802f38-c014-4dab-b769-b47cc86f443e">

## Project Intro/Objective

Yelp is a crowd-sourced app that allows users to critique restaurants by providing a star rating from 1-5 and a message arguing their reasons. It acts as a platform for restaurants to market their popularity, facilitate business-customer interaction, set standard expectations, and learn from customer criticism.

Goals of project

1. Use NLP to analyze specific text relating to positive and negative sentiment from Yelp reviews.
2. Develop a web application for stakeholders to visualize the KPI of their restaurants.
3. Implement an analysis tool from Restaurants' Positives and Criticisms.
4. Create a classification model that predicts the sentiment of user-inputted reviews.

## Business Applications of project

**Market Segmentation** - By analyzing the data and keywords, it's important for restaurant owners to focus on these characteristics when segmenting toward a specific audience.
Main focuses of restaurants:

1. Food (Specialities)
2. Service (Customer oriented)
3. Culture Identity - Korean, Mexican, Italian, etc
4. Time of day - Breakfast, Lunch, Dinner

Number 1) and 2) serve as a baseline/foundation for a successful restaurant. Number 3) and 4) serves as more of the restaurant's brand image, identity, and distinct characteristics that set it apart from other restaurants

**Self-awareness of a company** - KPI allows restaurant owners to manage their restaurant(s) and to benchmark performances through customer reviews, KPI metrics, and Sentiment Analysis.
Example: The Owner of a Korean Barbeque chain has 5 restaurants scattered around California.

- The owner can easily access each of the five restaurants from different cities.
- Can identify which restaurant has the best rating and most reviews.
- Similarities and differences of reviews from each restaurant.
- Based on reading reviews and sentiment analysis, the owner can deduce issues within the restaurant, whether it be personal, food, or service issues.

### Methods Used

- Machine Learning
- Data Visualization
- NLP
- etc.

### Technologies

- Python
- MongoDB
- Packages: Pandas, Numpy, Nltk, Textblob, Matplot, Plotly, WordCloud
- Jupyter
- Streamlit
- etc.

# Project Description

## 1) Data Ingestion - from collections.py

Started off by downloading the data from the Yelps open database.

yelp_academic_dataset_business.json

- Business_id
- name
- address
- city
- state
- latitude
- longitude
- stars
- review_count
- attributes
- categories

yelp_academic_dataset_review.json

- business_id
- text
- review_rating

Process

1. Create two separate data frames extracted from the business.json file and the review.json file
2. Join both DataFrames via business_id
3. Return the Dataframe to CSV

For the sake of runtime and size of the Yelp dataset (Over 6,990,280 million records and 150,346 businesses), I've extracted only 100 businesses from the JSON script and joined its customer reviews to them.

The total shape of the data frame is (8542,12)
There are 12 columns and 8542 observations

## 2) EDA

**Key Features:**
The main feature that we are mainly looking into is Text, which consists of an entire passage of the customer's review and its relationship with our featured engineered variable: Sentiment

- Text
- Customer_rating
- Emotion/Sentiment (Featured Engineered)

<img width="482" alt="image" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/5579e472-8938-43bf-a454-859e63e8a225">

Here is a count distribution of the sentiments.

Positive: 6823 count
Negative: 1719 count

Overall, 25.19% are Negative, and 74.81% are Positive. There is a case of class imbalances, and compensating for the balances may be challenging since the text data is unique and can't be synthetically created. It's best to preserve the integrity of the data and use model evaluation techniques such as a Stratified cross-fold validation to have an even amount of class distribution between each fold. For evaluation, looking at other performance metrics such as F1-Score or AUC-ROC may be best.

<img width="599" alt="image" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/4e4c7f1e-ac96-4845-ab88-d1e393b8f520">

From taking the distribution of the customer_ratings, we can see how there is a J-shaped curve.

**Positive Sentiment**

<img width="460" alt="Screenshot 2023-08-24 at 9 08 37 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/74bd0e77-f0a1-4b14-9777-97db69ccce5a">

<img width="461" alt="Screenshot 2023-08-24 at 9 08 17 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/f3173dbe-df84-4bf7-86ec-7875ee779184">



**Negative Sentiment**

<img width="449" alt="Screenshot 2023-08-24 at 9 09 08 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/4b490332-3b69-4030-bd38-1b5941b8f93b">


<img width="442" alt="Screenshot 2023-08-24 at 9 09 32 PM" src="https://github.com/Dionisio1013/Yelp-KPI-Metric/assets/106797659/afa1804f-b29d-4ae3-9173-fb54b51a44c3">



**Potential Biases**

1. Extreme Reviews: This type of J-curve represents when there is an initial decline followed by a sharp upward trend. This can represent that people are more inclined to be extreme with exceptional reviews and terrible reviews. In the case of 1-star reviews, one bad experience can cause a consumer to sway their opinions and promote hate. It's the baseline goal for all restaurants to avoid these instances and the ambitious goal for owners to find techniques to make consumers become positively extreme.

2. Missing information: Trying to break it down from the consumer's perspective. Many individuals may opt out of posting a good or bad review. This potential information gap may not be retrieved due to inactive users or individuals without exposure to Yelp.

3. Synthetic Data: In large metropolitan areas and higher competition, bots may create fake reviews from both positive and negative reviews.

**Preprocessing data**
To use NLP techniques it's best to use NLP techniques generally to work with tokenized, cleaned, and standardized data.

- Lowercased entire text
- Removed stop words
- Removed all punctuations
- Tokenizing (splitting each word of a sentence into its own token)
- Part of speech

**Feature Engineering**
Added a new feature where customer reviews that are rated 3,4,5 are "Positive" sentiment and 1,2 are "Negative" sentiment.

Analysis

No Sentiment reviews:
The most frequent keywords (top 5) of restaurants are the food served culture's names.

Positive reviews:
The most popular adjectives used for positive ratings are: "Good" and "Great". The words largely associated with these adjectives are:

- Place
- Food
- Service
- Time

Negative reviews:

## 3) Building the web application

Built using streamlit. I was able to create a web application that showcases KPIs of restaurants. It contains information regarding
a description of the business, sentiment analytics of words from customers, and a place to access customer reviews.

Description of restaurant

- Name and common hashtags
- Opening hours
- Address and Visual Location of Restaurant
- Total Reviews and Count of Positive and Negative Ratings

Sentiment Analytics

- Reviews most common keywords of customer ratings
- Most common adjectives from positive ratings
- Most common adjectives from negative ratings
- Word web to see which adjective links to what noun

Customer Reviews

- Slider that filters customer review ratings from 1-5
- Displays the customer's review ratings where you can double-click to expand.

## Getting Started

1. Web Application is [here](https://dionisio1013-hope-test-bdpevq.streamlit.app/).
2. Raw Data that has been used is called yelp20Revampedbusiness.csv & yelp40Revampedbusiness.csv.
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. Represents the pipeline from JSON to CSV with data of business,
5. yelpanalytics.py represents analytical findings from the Yelp Data
6. test.py represents web application
