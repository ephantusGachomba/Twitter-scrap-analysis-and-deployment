import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import collections
import re

#import panel as pn
#pn.extension('tabulator')
#import hvplot.pandas
import dataframe_image as dfi

import tweepy
import tweepy as tw
from textblob import TextBlob
#import pygal
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer 
from nltk import bigrams
from textblob import TextBlob
import networkx # for creating networknodes
import networkx as nx
#from pandas.io.json import json_normalize

import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

sns.set(font_scale=1.5)
sns.set_style("whitegrid")

api_key = "CWW5rvfohMPrJaf4KuACiE0FN"
api_key_secret = "r6iRyGGXzY9iwQaxjAjA64WweBIUFZnPM6tVpVfUMIrx3zlGtq"

access_token = "3403682451-rwoiYIUkNtEeNDZ97ZoRzbI5daYm9vOKRnzXlan"
access_token_secret = "QdrHp0YVqfIdVpyZVhTgIUSnGsOnL24sWEQUqUdbhUxVK"

#authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)




st.title("Twitter Sentiment Analysis and Deployment")
st.image("data:image/jpeg;base64,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", width = 500)


#Getting the keyword to analyse
keywords = st.text_input("Enter the keyword")
st.write(keywords)
limit = 1000 #Number of tweets to obtain


#limit at a time , we get 200.To solve this issue run code below.
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=200, tweet_mode = 'extended', wait_on_rate_limit=True).items(limit)

#Create DataFrame
columns = ['created_at', 'text', 'User']#, 'reply_count',	'retweet_count']#, 'user/location', 'user/followers_count']
data = []

for tweet in tweets:
  data.append([ tweet.created_at, tweet.full_text, tweet.user.screen_name])#, tweet.reply_count, tweet.retweet_count])# tweet.user.location, tweet.user.followers_count])

df = pd.DataFrame(data, columns=columns)
#df.rename(columns={"full_text": "text"}, inplace=True)

#df.head(5)

#Removing the duplicates
df = df.drop_duplicates()

#Convert the created_at column to datetime datatype
df['created_at'] = df['created_at'].astype('datetime64[ns]')

#Checking date ranges and hours
#Creating a column for hour
df['hour'] = df['created_at'].dt.hour
#Creating a column for days
df['date'] = df['created_at'].dt.date
#Creating a column for month
df['month'] = df['created_at'].dt.month
df.head()

# time series showing when the tweets for this analysis was created
reactions = df.groupby(['date']).count()
ax = reactions.text.plot(figsize=(15,6),ls='--',c='blue')
plt.ylabel('The Count of tweets collected')
plt.title('A Trend on the counts of tweets and the dates created' , fontsize=15, color= 'brown', fontweight='bold')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
st.pyplot()

# time series plot for the most active hours for tweeting
reactions = df.groupby(['hour']).count().sort_values(by='created_at',ascending=0)
reactions = df.groupby(['hour']).count()
ax = reactions.text.plot(figsize=(15,6),ls='--',c='green')
plt.ylabel('The Count of tweets collected')
plt.title('A Trend on the counts of tweets and the hours created',  fontsize=15, color= 'green', fontweight='bold')
ax.xaxis.grid(True)
ax.yaxis.grid(True)
st.pyplot()

##EXPLORATORY DATA ANALYSIS
#Creating a copy for the text column This will enable us work with the text column solely
df_tweets = df[['text']].copy()
#Dropping the duplicates
df_tweets = df_tweets.drop_duplicates()


## Text Processing
#A Function for cleaning the file (The text column in it)
def text_clean(df_tweets):
  #Lowercasing all the letters
  df_tweets['text'] = df_tweets['text'].str.lower() 

  #Removes mentions containing rt word
  df_tweets['text'] = df_tweets['text'].str.replace(r'rt @[A-Za-z0-9_]+:', '', regex=True) 
  #Removes mention just containing @word only
  df_tweets['text'] = df_tweets['text'].str.replace(r'@[A-Za-z0-9_]+', '', regex=True) 
  #Removing #tags 
  #df_tweets['text'] = df_tweets['text'].str.replace(r'#[A-Za-z0-9_]+', '', regex=True)  

  #Removing links
  df_tweets['text'] = df_tweets['text'].str.replace(r'http\S+', '', regex=True)
  df_tweets['text'] = df_tweets['text'].str.replace(r'www.\S+', '', regex=True) 

  #Removing punctuations and replacing with a single space
  df_tweets['text'] = df_tweets['text'].str.replace(r'[()!?]', ' ', regex=True)  
  df_tweets['text'] = df_tweets['text'].str.replace(r'\[.*?\]', ' ', regex=True)

  #Filtering non-alphanumeric characters
  df_tweets['text'] = df_tweets['text'].str.replace(r'[^a-z0-9]', ' ', regex=True) 

  #Removing Stoping words + keywords_to_hear
  stop = stopwords.words(['english', 'spanish', 'portuguese']) + ['l','pez', 'n', 'andr','p', 'si','est', 'c', 'qu']
  df_tweets['tweet_without_stopwords'] = df_tweets['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

text_clean(df_tweets)
# tokenize the tweets
df_tweets['tokenized_sents'] = df_tweets.apply(lambda row: nltk.word_tokenize(row['tweet_without_stopwords']), axis=1)


### Visualizing/InfoGraphics the text column (Unigram)
# Create a list of lists containing words for each tweet
words_in_tweet = list(df_tweets['tokenized_sents'])

#Calculate word frequencies
# List of all words across tweets
all_words = list(itertools.chain(*words_in_tweet))

# Create counter
counts_words = collections.Counter(all_words)

# transform the list into a pandas dataframe
df_counts_words = pd.DataFrame(counts_words.most_common(15), columns=['words', 'count'])

#A horizontal bar graph to visualize the most common words
fig, ax = plt.subplots(figsize=(10, 8))
# Plot horizontal bar graph
df_counts_words.sort_values(by='count').plot.barh(x='words',y='count',ax=ax,color="green")
ax.set_title("Common Words Found in Tweets ",  fontsize=15, color= 'violet', fontweight='bold')
#plt.savefig('count_unigram.png')
st.pyplot()

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wordcloud2 = WordCloud(background_color="white", max_words=100,
                       height=3000, width=3000,
                       colormap='Set2',
                       collocations=False,
                       repeat=True).generate(' '.join(df_counts_words["words"]))
# Generate plot
plt.figure(figsize=(10,7), facecolor='k')
plt.tight_layout(pad=0)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
#plt.savefig('cloud_uni.png')
st.pyplot()
#plt.show()


## Collection of Words – Bigrams
#Create a list of tokenized_sents
tweets_words = list(df_tweets['tokenized_sents'])
#Remove any empty lists
tweets_words_new = [x for x in tweets_words if x != []]
# Create list of lists containing bigrams in tweets
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_words_new]
# Flatten list of bigrams in clean tweets
bigrams = list(itertools.chain(*terms_bigram))
# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)
#Creating a dataframe of the most common bigrams
bigram_df = pd.DataFrame(bigram_counts.most_common(20),columns=['bigram', 'count'])


##Visualize Networks of Bigrams
# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')
# Create network plot 
G = nx.Graph()
# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(G, k=2)
# Plot networks
nx.draw_networkx(G, pos,font_size=16,width=3,edge_color='red',node_color='black',with_labels = False,ax=ax)
# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='aqua', alpha=0.55),
            horizontalalignment='center', fontsize=20)
plt.title('Visualize Networks of Bigrams',  fontsize=15, color= 'indigo', fontweight='bold')  
st.pyplot()
#plt.savefig('bigrams_network.png')
#plt.show()    


##Polarity
#Function to get the subjectivity Subjectivity refers to an individual's feelings, opinions, or preferences.
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity
#Create a function to get the polarity (Tells how positive or negative the text is)
def getPolarity(text):
  return TextBlob(text).sentiment.polarity
#Create two new columns
df['Subjectivity'] = df['text'].apply(getSubjectivity)
df['Polarity'] = df['text'].apply(getPolarity)

#plot the WordCloud
allwords  = ' '.join([txts for txts in df['text']])
wordCloud3 = WordCloud(background_color="white",
                      width = 5000, height = 3000, 
                      collocations=False,
                      colormap='Set2',
                      random_state = 1, repeat=True).generate(allwords)

plt.figure(figsize=(10,7), facecolor='k')
plt.imshow(wordCloud3, interpolation='bilinear')
plt.axis('off')
st.pyplot()
#plt.savefig('cloud_all_pol.png')
#plt.show()

#Create fxn to compute negative , neutral and positive analysis
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'
df['Analysis'] = df['Polarity'].apply(getAnalysis)
sortedDF = df.sort_values(by='Polarity')

#Plot the polarity and subjectivity
plt.figure(figsize=(28,10))
upper_limit = sortedDF.shape[0]

for i in range(0, upper_limit):
    #The range is the number of rows in our dataset
    plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='black')
plt.title("Sentiment Analysis Distribution",  fontsize=15, color= 'grey', fontweight='bold')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
st.pyplot()
#plt.savefig('sentiment_analysis_distribution.png')
#plt.show()


#Show the Value counts
sns.countplot(x='Analysis', data=df)
#plot and visualize the counts
sns.set(rc={'figure.figsize':(5,5)})
plt.title('Sentiment Analysis', fontsize=15, color= 'orange', fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
st.pyplot()
#plt.savefig('sentiment_analysis_graph.png')
#plt.show()



## polarity ( positive, negative , and neutral scores for each tweet)
'''using polarity_scores() we, 
will find all the positive, negative, and neutral scores for each tweet.'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
for i in range(df['text'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(df['text'][i])["compound"]
    pos = analyzer.polarity_scores(df['text'][i])["pos"]
    neu = analyzer.polarity_scores(df['text'][i])["neu"]
    neg = analyzer.polarity_scores(df['text'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })
    
#Converting the scores dictionary containing the scores into the data frame, then join the sentiments_score data frame with the df data frame.
sentiments_score = pd.DataFrame.from_dict(scores)
df = df.join(sentiments_score)

#sentiment_analysis_polarity_score
fig, ax = plt.subplots(figsize=(8, 6))
df.hist(column='Polarity', bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
        ax=ax, color="purple")
ax.set_title("Sentiments from Tweets(Using Polarity Score) on the tweets", fontsize=15, color= 'blue', fontweight='bold');
st.pyplot()
#plt.savefig('sentiment_analysis_polarity_score.png')
#plt.show()

#Finding the percentages of +ve, -ve and neutral
#Calculating the percentages
def percentage_polarity(part, whole_data):
  percentage = 100 * float(part) / float(whole_data)
  return round(percentage, 1)

negative = 0
positive = 0
neutral = 0

for index, row in df.iterrows():
  neg = row['Negative']
  pos = row['Positive']
  if neg > pos :
    negative += 1
    negative_list.append(df.text)
  elif pos > neg :
    positive += 1
  elif pos == neg:
    neutral += 1
positive_percentage = percentage_polarity(positive, df.shape[0])
negative_percentage = percentage_polarity(negative, df.shape[0])
neutral_percentage = percentage_polarity(neutral, df.shape[0])

#Creating PieCart for percentages
labels = ['Positive ['+str(positive_percentage)+'%]' , 'Neutral ['+str(neutral_percentage)+'%]','Negative ['+str(negative_percentage)+'%]']
sizes = [positive_percentage, neutral_percentage, negative_percentage]
colors = ['black', 'green','red']
my_circle=plt.Circle( (0,0), 0.5, color='white')
patches, texts = plt.pie(sizes,colors=colors, startangle=90)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result ", fontsize=15, color= 'aqua', fontweight='bold' )
plt.axis('equal')
st.pyplot()
#plt.savefig('sentiment_analysis_doughnut.png')
#plt.show()


## plotting wordcloud for positive, neutral and negative
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='snow',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='jet',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");
word_cloud(df['text'],)
st.pyplot()
#plt.savefig('cloud_all_2_pol.png' )

#Negative sentiment word cloud
word_cloud(df['text'][df['Positive'] < df['Negative']])
st.pyplot()
#plt.savefig('cloud_neg_pol.png')

#Positive sentiment word cloud
word_cloud(df['text'][df['Positive'] > df['Negative']])
st.pyplot()
#plt.savefig('cloud_pos_pol.png')

#Neutral cloud
word_cloud(df['text'][df['Positive'] == df['Negative']])
st.pyplot()
#plt.savefig('cloud_neu_pol.png')
