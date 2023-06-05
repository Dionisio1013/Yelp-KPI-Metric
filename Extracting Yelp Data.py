#!/usr/bin/env python
# coding: utf-8

# In[34]:


import requests
import json
import pymongo as py
import numpy as np
import pandas as pd
import nltk
from pymongo import MongoClient
from tqdm import tqdm


# In[2]:


# Connecting to MongoDB to be able to create collections on Jupyter Notebook

from pymongo.mongo_client import MongoClient

uri = "PUT YOUR MONGO CLIENT LINK HERE"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# In[29]:


# For the following blocks of code I wanted to connect to Yelp's API but there were so many limitations 
# in that you can only access three reviews of a restaurant and can only retrieve a small amount of restaurant
# businesses as well


# search_url = 'https://api.yelp.com/v3/businesses/search'
# reviews_url = 'https://api.yelp.com/v3/businesses/{}/reviews'
# api_key = 'sFH01x1QaKwY_BRM0b2KRKyMH-8pEkXymSZpz480RFsRcU_2jKyoM-UFveTTRarV5AdJDHB1FJH_TOUxi8Fs1Cr9F-JddRX0i0ZqOTunwaYwUo5Xt2KAGxW2AqpWZHYx'
# endpoint = 'https://api.yelp.com/v3'


# In[4]:


# import requests
# from pymongo import MongoClient

# # Yelp API key
# search_url = 'https://api.yelp.com/v3/businesses/search'
# reviews_url = 'https://api.yelp.com/v3/businesses/{}/reviews'
# api_key = 'API KEY'
# endpoint = 'https://api.yelp.com/v3'

# # Headers for the Yelp API request (include the API key)
# headers = {
#     'Authorization': f'Bearer {api_key}',
# }

# # Parameters for the Yelp API request (e.g., search for restaurants in San Francisco)
# params = {
#     'term': 'restaurants',
#     'location': 'Los Angeles',
#     'limit': 50,
# }

# # Step 1: Make an HTTP request to the Yelp API to retrieve business data
# response = requests.get(search_url, headers=headers, params=params)
# businesses = response.json()['businesses']



# In[ ]:


# Connecting to host 
client = MongoClient('mongodb://localhost:27017/')
db = client['yelp_db']


# In[5]:


# Retrieving data regarding the business aspect of the restaurants

# Reading the JSON script and using a for loop to append each line into the business array.
business = []
with open('yelp_academic_dataset_business.json', 'r') as file:
    for line in tqdm(file, desc='Extracting businesses'):
        x = json.loads(line)
        # Have to implement this code in order for the JSON script to pass schema validation
        if x is not None and 'categories' in x and x['categories'] is not None and 'Restaurants' in x['categories']:
            business.append(x)


# In[15]:


# For scalability we will start off with only 100 businesses for analyzation and KPI prototype
scalebusiness = business[0:100]


# In[7]:


# Reading the JSON script containing customer reviews

reviews = []
with open('yelp_academic_dataset_review.json', 'r') as file:
    for line in tqdm(file, desc='Extracting reviews'):
        x = json.loads(line)
        reviews.append(x)
        


# In[8]:


business


# In[ ]:


reviews


# In[9]:


## Created a Schema Validation for information regarding the restaurant/business
businessvalidator = {
    "$jsonSchema": {
        "bsonType": "object", 
        "required": ["name", "city", "business_id"],
        "properties": {
            "name": {
                "bsonType": "string",
                "description": "must be a string and required"
            },
            "stars": {
                "bsonType": "number",
                "description": "must be a number and required"
            },
            "categories": {
                "bsonType": "string",
                "description": "must be an string and required"
            },
             "city": {
                "bsonType": "string",
                "description": "must be a string and required"
             },
            "business_id": {
                "bsonType": "string",
                "description": "must be a string and required"
            }
        }
    }
}


# In[10]:


# Created schema validaiton for reviewers. Important to have text, stars, and business_id in order to join both 
# collections
reviewersvalidator = {
    "$jsonSchema": {
        "bsonType": "object", 
        "required": ["text", "stars", "business_id"],
        "properties": {
            "text": {
                "bsonType": "string",
                "description": "must be a string and required"
            },
            "stars": {
                "bsonType": "number",
                "description": "must be a number and required"
            },
            "business_id": {
                "bsonType": "string",
                "description": "must be a string and required"
            },
            
        }
    }
}


# In[16]:


# dropping previous collections and creating two new collections
db.owners.drop()
db.reviewers.drop()
db.create_collection('owners')
db.create_collection('reviewers')


# In[17]:


# implementing schema validation
db.command("collMod", "owners", validator=businessvalidator)
db.command("collMod", "reviewers", validator=reviewersvalidator)


# In[18]:


from tqdm import tqdm

business_ids = set()  # Set to store business IDs

# Iterate through scalebusiness
for restaurants in tqdm(scalebusiness, desc='Inserting businesses'):
    # Insert basic business information into the 'businesses/owners collection' collection
    db.owners.insert_one(restaurants)

    # Store business ID in the set
    business_id = restaurants['business_id'].strip("'")
    business_ids.add(business_id)

# Iterate through reviews 
for review in tqdm(reviews, desc='Inserting reviews'):
    business_id = review['business_id'].strip("'")
    # This will gather all the reviews that is under the 100 businesses
    # Joined by a business_id
    if business_id in business_ids:
        db.reviewers.insert_one(review)


# In[21]:


owners1 = []

# Converting the owners collection into a dataframe for analyzation

for document in db.owners.find({}):
    print(document)
    print("")
    name = document['name']
    restaurant_rating = document['stars']
    city = document['city']
    business_id = document['business_id']
    longitude = document['longitude']
    latitude = document['latitude']
    categories = document['categories']
    hours = document['hours']
    address = document['address']
    state = document['state']
    
    owners1.append([name,city,restaurant_rating, business_id, longitude, latitude,categories,hours, address, state])
    
owners = pd.DataFrame(owners1,columns = ['name','city','restaurant_rating', 'business_id','longitude', 'latitude','categories','hours', 'address', 'state'])


# In[22]:


owners


# In[23]:


reviews1 = []

# Creating a dataframe for customer reviews collection

for document in db.reviewers.find({}):
    customer_rating = document['stars']
    text = document['text']
    business_id = document['business_id']
    reviews1.append([customer_rating,text,business_id])
    
reviews = pd.DataFrame(reviews1, columns = ['customer_rating','text','business_id'])


# In[24]:


# Merging the two dataframes on business_id to join data
yelpdf = reviews.merge(owners, how = 'inner', on = 'business_id')
yelpdf.head(5)


# In[25]:


# Converting the data into a csv to Data Exploration
yelpdf.to_csv('yelp100Revampedbusiness.csv')


# In[26]:


yelpdf

