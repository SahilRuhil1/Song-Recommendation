#!/usr/bin/env python
# coding: utf-8

# 
# # 1. Data Exploration:

# In[33]:


# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


# Reading the dataset

df = pd.read_csv(r"C:\Users\sahil\Desktop\excelr data science\Projects\Song Recommendation\song_recommendation_dataset.csv")


# In[3]:


# Displaying the head of dataset

df.head()


# In[4]:


# Displaying the tail of dataset

df.tail()


# In[5]:


# General info of dataset

df.info()


# In[6]:


# Converting datatypes into suitable format

# Convert 'Duration' from milliseconds to seconds
df['Duration'] = df['Duration'] / 1000

# Convert 'Timestamp' to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Convert Release Year to Decade
df['Decade'] = (df['Release_Year'] // 10) * 10


# In[7]:


# Dropping unnecessary columns

df.drop(columns=['Song_ID'], inplace=True)


# In[8]:


# Checking presence of null values

df.isnull().sum()


# In[9]:


# Displaying shape of dataset 

df.shape


# In[10]:


# Checking presence of duplicate values

df.duplicated().sum()


# In[11]:


# Summary statistics

df.describe()


# # 2. Exploratory Data Analysis:

# In[12]:


# Distribution of song popularity

plt.figure(figsize=(10,5))
sns.histplot(df['Popularity'], bins=30, kde=True)
plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))
sns.countplot(y=df['Genre'], order=df['Genre'].value_counts().index, palette="coolwarm")
plt.title("Number of Songs per Genre")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()


# In[14]:


plt.figure(figsize=(12,6))
sns.boxplot(x='Genre', y='User_Rating', data=df, palette="Set3")
plt.xticks(rotation=45)
plt.title("Distribution of User Ratings by Genre")
plt.show()


# In[15]:


df['Release_Year'] = df['Release_Year'].astype(int)

plt.figure(figsize=(12,6))
sns.lineplot(x="Release_Year", y="Popularity", data=df, ci=None, marker="o")
plt.title("Trend of Song Popularity Over the Years")
plt.xlabel("Release Year")
plt.ylabel("Average Popularity")
plt.grid()
plt.show()


# In[16]:


# Plotting top 10 artists based on popualrity

# Aggregate popularity per artist
artist_popularity = df.groupby('Artist')['Popularity'].sum().reset_index()

# Get top 10 artists based on total popularity
top_artists = artist_popularity.sort_values(by="Popularity", ascending=False).head(10)

# Plot the top 10 artists based on popularity
plt.figure(figsize=(12, 6))
sns.barplot(x=top_artists["Popularity"], y=top_artists["Artist"], palette="magma")
plt.xlabel("Total Popularity")
plt.ylabel("Artist")
plt.title("Top 10 Artists by Popularity")
plt.show()


# In[17]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[18]:


df['YearMonth'] = df['Timestamp'].dt.to_period('M')

plt.figure(figsize=(12,6))
df.groupby('YearMonth').size().plot(kind='line', marker='o', color='b')
plt.title("User Engagement Over Time")
plt.xlabel("Time (Year-Month)")
plt.ylabel("Number of Interactions")
plt.xticks(rotation=45)
plt.grid()
plt.show()


# In[19]:


sns.pairplot(df[['Popularity', 'User_Rating', 'Duration', 'Release_Year']], diag_kind='kde')
plt.show()


# In[20]:


plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Duration"], y=df["Popularity"], alpha=0.7, hue=df["Genre"], palette="coolwarm")
plt.xlabel("Duration (seconds)")
plt.ylabel("Popularity")
plt.title("Song Duration vs Popularity")
plt.show()


# In[21]:


plt.figure(figsize=(12,6))
sns.barplot(x=df.groupby("Genre")["Popularity"].mean().sort_values(ascending=False).index, 
            y=df.groupby("Genre")["Popularity"].mean().sort_values(ascending=False).values, 
            palette="viridis")
plt.xlabel("Genre")
plt.ylabel("Average Popularity")
plt.xticks(rotation=45)
plt.title("Average Popularity by Genre")
plt.show()


# In[22]:


plt.figure(figsize=(12,6))
sns.countplot(x=df["Decade"], palette="rocket")
plt.xlabel("Decade")
plt.ylabel("Number of Songs")
plt.title("Number of Songs Released Per Decade")
plt.show()


# In[23]:


# Boxplot to detect outliers

# List of numerical columns to check for outliers
num_features = ["Popularity", "User_Rating", "Duration"]

# Create boxplots for detecting outliers
plt.figure(figsize=(12,6))
for i, col in enumerate(num_features, 1):
    plt.subplot(1, len(num_features), i)
    sns.boxplot(y=df[col], color="lightblue")
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


# # 3. Model Building
# 

# Now we can build model using techniques, first is using content based filtering and second using collaborative filtering

# # 1) Content Based Filtering:

# In[24]:


# Normalize numerical features

scaler = MinMaxScaler()
df[['Popularity', 'Duration', 'User_Rating']] = scaler.fit_transform(df[['Popularity', 'Duration', 'User_Rating']])


# In[25]:


# Create a combined text feature for content-based filtering

df['Combined_Features'] = df['Artist'] + " " + df['Genre']


# In[26]:


# TF-IDF Vectorization for text-based features
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['Combined_Features'])


# In[27]:


# Compute Cosine Similarity
cosine_sim = cosine_similarity(feature_matrix, feature_matrix)


# In[28]:


# Function for Content-Based Filtering
def recommend_songs(song_name, df, cosine_sim, top_n=10):
    if song_name not in df['Song_Name'].values:
        return "Song not found in the dataset!"
    
    # Get song index
    idx = df[df['Song_Name'] == song_name].index[0]
    
    # Get similarity scores for all songs
    similarity_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort songs based on similarity
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar songs (excluding itself)
    recommended_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    
    # Return recommended songs
    return df.iloc[recommended_indices][['Song_Name', 'Artist', 'Genre', 'Popularity']]


# # 2) Collaborative Based Filtering 

# In[29]:


# Collaborative Filtering Using User-Based Similarity
user_song_matrix = df.pivot_table(index='User_ID', columns='Song_Name', values='User_Rating').fillna(0)


# In[30]:


# Compute Cosine Similarity Between Users
user_similarity = cosine_similarity(user_song_matrix)
user_sim_df = pd.DataFrame(user_similarity, index=user_song_matrix.index, columns=user_song_matrix.index)


# In[31]:


# Function for User-Based Collaborative Filtering
def user_based_recommendation(user_id, df, user_sim_df, top_n=5):
    if user_id not in user_sim_df.index:
        return "User not found!"
    
    # Find similar users
    similar_users = user_sim_df[user_id].sort_values(ascending=False).index[1:top_n+1]
    
    # Get songs rated highly by similar users
    song_recommendations = df[df['User_ID'].isin(similar_users)].groupby('Song_Name')['User_Rating'].mean().sort_values(ascending=False).head(top_n)
    
    return song_recommendations.reset_index()


# In[34]:


# Streamlit App Interface
st.title('Song Recommendation System')

# Option to choose between Content-Based or Collaborative Filtering
option = st.selectbox(
    'Choose Recommendation Method:',
    ['Content-Based Filtering', 'Collaborative Filtering']
)

if option == 'Content-Based Filtering':
    st.header("Content-Based Recommendations")
    
    # Input song name for recommendations
    song_name = st.text_input("Enter a Song Name:", "")
    
    if song_name:
        recommendations = recommend_songs(song_name, df, cosine_sim)
        if isinstance(recommendations, str):
            st.write(recommendations)  # If song not found
        else:
            st.write("Recommended Songs:")
            st.dataframe(recommendations)

elif option == 'Collaborative Filtering':
    st.header("Collaborative Filtering Recommendations")
    
    # Input user ID for recommendations
    user_id = st.text_input("Enter User ID:", "")
    
    if user_id:
        recommendations = user_based_recommendation(user_id, df, user_sim_df)
        if isinstance(recommendations, str):
            st.write(recommendations)  # If user not found
        else:
            st.write("Recommended Songs for User:")
            st.dataframe(recommendations)


# In[ ]:




