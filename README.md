# Yelp KPI Web Application & Sentiment Analysis

Here is a link to my [web application!](https://dionisio1013-hope-test-bdpevq.streamlit.app/)
This project is a part of the [Yelp Dataset](https://www.yelp.com/dataset).

#### -- Project Status: [Active]

## Project Intro/Objective
Yelp is a crowd-source app that allows users to critique restaurants by providing
a star rating from 1-5 and a message argumenting their reasons. It acts as a way for 
restaurants to market their popularity, have business-customer interaction, set standards expectations
and to able to lean from customer criticism. 

Goals of project
1) Dissect why certain restaurants have higher ratings from each other and see what distinguishes from each other.
2) The goal is to use natural language processors as a way to make predictions on why some ratings score 
higher than others. We want to learn from unhappy customers.
3) Develop a web application for stakeholders to use to pursue data driven decision making from my findings

## Business Applications of project Abstract
Market Segmentaiton - The pillars of a Restaurant aligns from 

Common adjectives and links to Positive and Negative Reviews


### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* NLP
* Predictive Modeling
* etc.

### Technologies
* Python
* MongoDB
* Packages: Pandas, Numpy, Nltk
* jupyter
* Streamlit
* etc. 

# Project Description

## 1) Getting the Data - from collections.py
Started off by downloading the data from the Yelps open database

I wanted to organize my data and be able to create a pipeline to get data from a JSON script, create two collections via MongoDB,
develop a schema validation, and then transfer the data from collection to csv for analytics and for web application. 

## 2) Analyzing the Data & Key Findings


## 3) Building the web application
Built using streamlit. I was able to create a web application that showcases KPIs of restaurants. It contains information regarding 
a description of the business, sentiment analytics of words from customers and a place to access customer reviews of.

Description of restaurant - 
- Name and common hashtags
- Opening hours
- Address and Visual Location of Restaurant
- Total Reviews and Count of Positive and Negative Ratings

Sentiment Analytics
- Reviews most common keywords of customer ratings
- Most common adjectives from positive ratings
- Most common adjectives from negative ratings
- Word web to see which adjective links to what noun

## Getting Started

1. Web Application is [here](https://dionisio1013-hope-test-bdpevq.streamlit.app/).
2. Raw Data that has been used is called yelp20Revampedbusiness.csv & yelp40Revampedbusiness.csv.
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. Represents the pipeline from JSON to CSV with data of business,
5. yelpanalytics.py represents analytical findings from the yelp Data
6. test.py represents webapplication
