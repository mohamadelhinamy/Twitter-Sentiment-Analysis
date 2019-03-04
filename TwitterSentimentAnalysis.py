# In[1]:


import csv as csv
import numpy as np 
with open('Tweets.csv') as csv_file:
    tweets = csv.reader(csv_file , delimiter=',')
    labels = list()
    airlines = list()
    sentiment_confidence = list()
    retweet_count = list()
    text = list()
    for tweet in tweets:
        labels.append(tweet[1])
        airlines.append(tweet[5])
        sentiment_confidence.append(tweet[2])
        retweet_count.append(tweet[9])
        text.append(tweet[10])
    print(len(labels))
    text_copy = text.copy()


# In[2]:


from langdetect import detect 
text2 = list()
airlines2 = list()
sentiment_confidence2 = list()
labels2= list()
retweet_count2 = list()
for i,t in enumerate(text[:]): 
    
    if detect(t) != 'en' :
        continue
    
    if len(t) < 20 :
        continue
    
    if retweet_count[i] != "0" :
        continue
                

    
    text2.append(text[i])
    airlines2.append(airlines[i])
    sentiment_confidence2.append(sentiment_confidence[i])
    labels2.append(labels[i])
    retweet_count2.append(retweet_count[i])
    
    
labels = labels2
print(len(labels))
text = text2
print(len(text))
airlines = airlines2
sentiment_confidence = sentiment_confidence2
retweet_count = retweet_count2
        


# In[4]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import re,string
from string import punctuation


text3 = list()
text4 = list()
text5 = list()
text6 = list()
text7 = list()
             
# removing urls
for t in text:
    r = re.sub(r"http\S+", "",t).strip()
    text3.append(r)
    
#removing punctuation
pattern = string.punctuation
for t in text3:
    r1 = ''.join([word for word in t if word not in pattern])
    text5.append(r1)
                
#tokenizing
for t in text5:
    tokenized = word_tokenize(t)
    text4.append(tokenized)

#stemming                
ps = PorterStemmer()
text6 = [[ps.stem(word) for word in words] for words in text4]
# print(text6[7])

#removing stopwords
for t in text6:
    t = [word for word in t if word not in stopwords.words('english')]
    text7.append(t)
# print(text7[7])    


text = [' '.join(words) for words in text7]
print(text[7])
print(len(text))
print(len(labels))


# In[8]:


# create training and testing vars
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
X_train, X_test, y_train, y_test = train_test_split(text, labels , test_size=0.2,random_state=1)
Vec = TfidfVectorizer()
X_train2 = Vec.fit_transform(X_train)
X_test2 = Vec.transform(X_test)
print (X_train2.shape)
print(X_test2.shape)


# In[9]:


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train2, y_train)
# print(NB.score(X_train2, y_train))


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train2, y_train)
# print(neigh.score(X_train2, y_train))


# In[11]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train2, y_train)
# print(RF.score(X_train2, y_train))


# In[12]:


from sklearn.metrics import f1_score
y_pred = RF.predict(X_test2)
score = f1_score(y_test, y_pred, average='micro')
print(score)


# In[13]:


from sklearn.metrics import f1_score
y_pred = neigh.predict(X_test2)
score = f1_score(y_test, y_pred, average='micro')
print(score)


# In[14]:


from sklearn.metrics import f1_score
y_pred = NB.predict(X_test2)
score = f1_score(y_test, y_pred, average='micro')
print(score)


# In[ ]:




