import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import streamlit as st
from streamlit_tags import st_tags
from wordcloud import WordCloud
from nltk.tag import pos_tag
from textblob import TextBlob
import openai



# First reading in the yelp file
yelpdf = pd.read_csv('yelp.csv')

yelpdf.drop(columns = 'Unnamed: 0', inplace = True)

noneditedyelpdf = yelpdf.copy()


# Preprocessing text data 
def preprocess(text):
    # Using regex to remove punctuations
    text = text.replace(r'[^\w\s]+', '', regex=True)
    # Lowercasing words 
    text = text.str.lower()
    return text

yelpdf['text'] = preprocess(yelpdf['text'])

# Implementing the removal of stop words 
stop_words = stopwords.words('english')
yelpdf['text'] = yelpdf['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

## THIS CODE IS FOR THE SIDE BAR of the Streamlit screen
st.sidebar.header("YELP RESTAURANT KPI")

# Creating the side bar select box where you can scroll through all the unique reestaurants
selected_name = st.sidebar.selectbox("Select a Restaurant:", yelpdf['name'].unique())
hashtagarray = []
for x in yelpdf[yelpdf['name']==selected_name]['categories'].head(1):
    x = x.split(',')

for hashtag in x:
    hashtag = hashtag.lstrip()
    hashtag = '#'+hashtag
    hashtagarray.append(hashtag)

# Creating the list of hashtags
options = st.sidebar.multiselect('List of Hashtags',hashtagarray,hashtagarray)

# Displaying the City, Address and hours of operations of the restaurant 
st.sidebar.header("Address:")
st.sidebar.write(yelpdf[yelpdf['name']==selected_name]['address'].head(1).values[0])
st.sidebar.write(yelpdf[yelpdf['name']==selected_name]['city'].head(1).values[0] + ', ' + yelpdf[yelpdf['name']==selected_name]['state'].head(1).values[0])
st.sidebar.header('Hours of Operation: ')

hoursdictionary = {}
if yelpdf[yelpdf['name']== selected_name]['hours'].head(1).isna().values[0] != True:
    for x in yelpdf[yelpdf['name']==selected_name]['hours'].head(1):
        x = x.replace('{','')
        x = x.replace('}','')
        x = x.replace("'",'')
        x = x.replace(",",'')
        hoursarray = x.split()

    for i in range(0, len(hoursarray),2):
        day = hoursarray[i].strip(':')
        time = hoursarray[i+1]
        hoursdictionary[day] = time
        
    df = pd.DataFrame(hoursdictionary.items(), columns = ['Day','Time'])
    df[['Start', 'End']] = df['Time'].str.split('-', expand=True)

    df['Start'] = pd.to_datetime(df['Start'], format='%H:%M').dt.strftime('%I:%M %p')
    df['End'] = pd.to_datetime(df['End'], format='%H:%M').dt.strftime('%I:%M %p')

    df = df.drop(columns = 'Time')

    for index, row in df.iterrows():
        st.sidebar.write((row['Day'] +': ' + row['Start'] + '-' + row['End']))
else:
        st.sidebar.write(None)


    




# Feature Engineering
yelpdf['sentiment']= ' '
yelpdf.loc[yelpdf['customer_rating'].isin([5.0,4.0,3.0]), 'emotions'] = 'positive'
yelpdf.loc[yelpdf['customer_rating'].isin([1.0,2.0]), 'emotions'] = 'negative'

Aggregatereviews = yelpdf.groupby('name').size().sort_values(ascending = False)

Over100 = yelpdf.copy()

st.markdown('# ' + selected_name)



### Star rating -- KPI

rating = float(Over100[Over100['name'] == selected_name]['restaurant_rating'].mean())
star_rating = int(rating) * ":star:"



total_reviews = int(Over100[Over100['name'] == selected_name].groupby('emotions').count().sum()[0])

# Positive and Negative
t = Over100[Over100['name'] == selected_name].groupby('emotions').count()

positive_reviews = t.loc['positive','sentiment']

try:
    negative_reviews = t.loc['negative', 'sentiment']
except KeyError:
    negative_reviews = 0

left_column, middle_columns, middle_right,right_column = st.columns(4)
with left_column:
    st.subheader("Total Stars:")
    st.warning(f"{rating} {star_rating}")
with middle_columns:
    st.subheader("Total Reviews")
    st.info(f"{total_reviews}")
with middle_right:
    st.subheader("Positive")
    st.success(f"{positive_reviews}")
with right_column:
    st.subheader("Negative")
    st.error(f"{negative_reviews}")





Positive_reviews = Over100[Over100['emotions'] == 'positive']
Neutral = Over100[Over100['emotions'] == 'neutral']
Negative_reviews = Over100[Over100['emotions'] == 'negative']

from collections import Counter
general_word_dict = {}


for name, group in Over100.groupby('name'):
    general_word_dict[name] = {}
    for word in group['text']:
        words = word.split()
        for x in words:
            if x in general_word_dict[name]:
                general_word_dict[name][x] += 1
            else:
                general_word_dict[name][x] = 1



### Positive WORD COUNTER ANALYSIS 


Positive_business = Positive_reviews[Positive_reviews['name'] == selected_name]
positivewordcount = {}

for index, row in Positive_business.iterrows():
    text = row['text']
    words = text.split()
    for word in words:
        if word not in positivewordcount:
            positivewordcount[word] = 1
        else:
            positivewordcount[word] += 1

datawords = {'word':positivewordcount.keys(), 'count': positivewordcount.values()}
positivewordsdf = pd.DataFrame(datawords)

positivewords = []

for index,row in positivewordsdf.iterrows():
        if TextBlob(row['word']).sentiment.polarity > 0:
            positivewords.append(row['word'])

positive_words_df = positivewordsdf[positivewordsdf['word'].isin(positivewords)]






### NEGATIVE WORD COUNTER ANALYSIS 


Negative_business = Negative_reviews[Negative_reviews['name'] == selected_name]
negativewordcount = {}

for index, row in Negative_business.iterrows():
    text = row['text']
    words = text.split()
    for word in words:
        if word not in negativewordcount:
            negativewordcount[word] = 1
        else:
            negativewordcount[word] += 1

datawords = {'word':negativewordcount.keys(), 'count': negativewordcount.values()}
negativewordsdf = pd.DataFrame(datawords)

negativewords = []


for index,row in negativewordsdf.iterrows():
        if TextBlob(row['word']).sentiment.polarity < 0:
            negativewords.append(row['word'])

negative_words_df = negativewordsdf[negativewordsdf['word'].isin(negativewords)]




# Creating a navigation select bar that switches the application 
selected_sentiment = st.selectbox("Select an Option:", ['Description of Restaurant', 'Sentiment Analysis', 'Reviews of Restaurant', 'GPT Predictor', 'Sentiment Classification Predictor'])

if selected_sentiment == 'Sentiment Analysis':

    genre = st.radio("Select Sentiment", ('Most Frequent (No Sentiment)','Positive', 'Negative'))

    if genre == 'Positive':
        st.write('Displayed are key words related to Positive Reviews.')

            ## POSITIVE SENTIMENT ANALYSIS 
        positive_word_frequency = positive_words_df.set_index('word').to_dict()['count']

        # Generate the word cloud
        wordcloud = WordCloud(background_color='white', scale=2).generate_from_frequencies(positive_word_frequency)

        # Display word cloud using Streamlit
        st.title('Word Cloud')
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        st.pyplot(fig)


        # Create a DataFrame for bar chart data
        top5words = positive_words_df.sort_values('count', ascending=False).head(5)
        plotlybarreverse = top5words.sort_values('count', ascending = True)

        ## Plotly Bar chart
        fig_word_count = px.bar(
            plotlybarreverse,
            x='count',
            y='word',
            orientation = 'h',
            title= '<b>Word by Frequency'
        )

       
        st.plotly_chart(fig_word_count)
    
    if genre == 'Most Frequent (No Sentiment)':
        st.write("Most frequent key words related to all Reviews.")
        wordcloud = WordCloud()

        wordcloud = WordCloud(background_color='white', scale=2).generate_from_frequencies(general_word_dict[selected_name])

        # Display word cloud using Streamlit
        st.title('Word Cloud')
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        st.pyplot(fig)


        # Create a DataFrame for bar chart data
        word = general_word_dict[selected_name].keys()
        values = general_word_dict[selected_name].values()
        data = {"word": word, "frequency": values}
        df = pd.DataFrame(data)
        top5words = df.sort_values('frequency', ascending=False).head(5)
        plotlybarreverse = top5words.sort_values('frequency', ascending = True)

        ## Plotly Bar chart
        fig_word_count = px.bar(
            plotlybarreverse,
            x='frequency',
            y='word',
            orientation = 'h',
            title= '<b>Word by Frequency'
        )

        st.plotly_chart(fig_word_count)

    if genre == 'Negative' and negative_reviews > 0: 
        st.write("Displayed are key words related to Negative Reviews.")
   
        negative_word_frequency = negative_words_df.set_index('word').to_dict()['count']

        # Generate the word cloud
        wordcloud = WordCloud(background_color='white', scale=2).generate_from_frequencies(negative_word_frequency)

        # Display word cloud using Streamlit
        st.title('Word Cloud')
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_axis_off()
        st.pyplot(fig)


        # Create a DataFrame for bar chart data
        top5words = negative_words_df.sort_values('count', ascending=False).head(5)
        plotlybarreverse = top5words.sort_values('count', ascending = True)

        ## Plotly Bar chart
        fig_word_count = px.bar(
            plotlybarreverse,
            x='count',
            y='word',
            orientation = 'h',
            title= '<b>Word by Frequency'
        )

        st.plotly_chart(fig_word_count)
    else:
        st.write('No Negative Reviews')


# GPT Predictor
secret = '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'

openai.api_key = secret
if selected_sentiment == 'GPT Predictor':
    st.write("Here is a GPT Predictor that will sample reviews and write an assessment regarding its performance")
    if st.button("Generate Assessments"):
        sampled_reviews = []
        for x in np.arange(1.0, 6.0, 1.0):
            ratings = Over100[(Over100['customer_rating'] == x)]['text']
            
            for num_samples in range(3, 0, -1):
                try:
                    sampled_rows = ratings.sample(n=num_samples, replace=False)
                    
                    for review in sampled_rows:
                        sampled_reviews.append(review)
                    break  # Exit the loop if successful sampling occurs
                except ValueError:
                    continue  # If there's an error (e.g., empty DataFrame or not enough data), try again
                    
        combined_text = ' '.join(sampled_reviews)

        prompt = 'Here is a list of reviews. Write me a summary on why this restaurant is good and bad. Can you narrow it down to three points for both good and bad.' + combined_text

        completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}]
        )

        st.write(completion.choices[0].message['content'])
    

if selected_sentiment == 'Description of Restaurant':
    st.header('Location:')
    
    data = pd.DataFrame({
    'latitude': [float(yelpdf[yelpdf['name']==selected_name]['latitude'].head(1))],
    'longitude': [float(yelpdf[yelpdf['name']==selected_name]['longitude'].head(1))]})
    st.map(data)



if selected_sentiment == 'Reviews of Restaurant':
    age = st.slider('Selected Customer Review Rating', 1, 5, 1)
    st.write("Selected Rating: ", age, ' stars.')

    noneditedyelpdf[(noneditedyelpdf['name'] == selected_name) & (noneditedyelpdf['customer_rating'] == age)][['text','customer_rating']]



## Creating the modeling 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score



if selected_sentiment == 'Sentiment Classification Predictor':

    # Splitting the X and y variables
    Xfeatures = yelpdf['text']
    y = yelpdf['emotions']

    # Splitting Training and Testing data
    X_train, X_test , y_train , y_test =  train_test_split(Xfeatures, y , test_size = 0.2)

    tfidf_vectorizer = TfidfVectorizer()

    # After splitting data we apply the fit_transform of x_train and then apply transform X_test to maintain relativity
    x_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Using these hyperparameters because they had the highest GridSearchCV
    SVCmodel = SVC(C = 1, gamma = 'scale', kernel = 'linear')

    # Fitting the final model
    SVCmodel.fit(x_tfidf, y_train)

    # Testing model on unseen data
    svm_predictions = SVCmodel.predict(X_test_tfidf)

    # Evaluate the model's performance via Precision, recall, f1-score, support
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_classification_report = classification_report(y_test, svm_predictions)
    
    st.write("Express your thoughts (good or bad) about a recent Restaurant you've been to!")

    user_review = st.text_input('Write a review', '')    
    user_review = tfidf_vectorizer.transform([user_review])
    
    # Get the decision values (raw scores) for each class
    decision_values = SVCmodel.decision_function(user_review)

    # Convert decision values to probability estimates using sigmoid function
    probabilities = 1 / (1 + np.exp(-decision_values))

    st.write("This is your review:", SVCmodel.predict(user_review), "This message had a probability of:", probabilities)

    st.write("This is the SVM model's accuracy", svm_accuracy)
    