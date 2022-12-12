import json
import numpy as np
from os import getcwd
import pandas as pd

import pickle
from sklearn import preprocessing
import re
from textblob import Word, TextBlob
from string import punctuation as pn
from nltk.stem.snowball import SnowballStemmer
from gensim.parsing.preprocessing import STOPWORDS


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn import  naive_bayes, pipeline
from sklearn.metrics import accuracy_score

# transform json file into dataframe form and export 
def set_path():
   abspath = getcwd()
   dname = os.path.dirname(abspath)
   os.chdir(dname)
set_path()

# Open raw data
trainjs = pd.read_json("data/raw/train.json")
trainjs.to_pickle("data/processed/dirty_df.pkl")
testjs = pd.read_json("data/raw/test.json")
testjs.to_pickle("data/processed/test_dirty_df.pkl")

# Cleaing text: 

def process_row(row):
   # Deleting email:
   row = re.sub('(\S+@\S+)(com|\s+com)', ' ', row)
   # Deleting username:
   row = re.sub('(\S+@\S)', ' ', row)
   # punctuation & lower case:
   punctuation = pn + '—“,”‘-’'
   row = ''.join(char.lower() for char in row if char not in punctuation)
   # Erasing stopword, converting plurals into singular, detach punctuation
   stop = STOPWORDS
   row = TextBlob(row)
   row = ' '.join(Word(word).lemmatize() for word in row.words if word not in stop)

   # Bring word to its root form
   stemmer = SnowballStemmer('english')
   row = ' '.join([stemmer.stem(word) for word in row.split() if len(word) > 2])
   # Erase extra white space
   row = re.sub('\s{1,}', ' ', row)

   return row

trainjs = pd.read_pickle("data/processed/dirty_df.pkl")

### Delete duplicate
df_cleaned=trainjs.drop([trainjs.index[10230],trainjs.index[12039],trainjs.index[11910],trainjs.index[10221],trainjs.index[10559],trainjs.index[7177],trainjs.index[5839],trainjs.index[2070], trainjs.index[2459],trainjs.index[2612]])

### Processing text, re-label author, drop unnessary columns
##### Processing texts
# Call cleaning functions

df_cleaned['title'] = df_cleaned['title'].apply(process_row)
df_cleaned['abstract'] = df_cleaned['abstract'].apply(process_row)

def mergingtext(df):
   full_content = []
   for i in range(len(df)):
      fulltext = df.iloc[i]['title'] + ' ' + df.iloc[i]['abstract']
      full_content.append(fulltext)
   df['content'] = full_content

mergingtext(df_cleaned)

auth_le = preprocessing.LabelEncoder()
authid_enc = auth_le.fit_transform(df_cleaned['authorId'])
df_cleaned['authId_enc'] = authid_enc
df_cleaned = df_cleaned[['content','authId_enc' ]].copy()
df_cleaned.reset_index(inplace=True, drop = True)
df_cleaned[:3]
# saving author label to true authorId
with open("code/authorIdlabel.pkl", 'wb') as f:
      pickle.dump(file=f, obj=auth_le)

### Spliting data
from collections import Counter
import random
# choosing random 1702 authors that have more than 3 encounters in the dataset for validation
count = Counter(df_cleaned['authId_enc'])
frequentAuthor = list({k:v for k,v in count.items() if count[k] >=3}.keys())
len(frequentAuthor)

random.shuffle(frequentAuthor)

val_id = []
for auth in frequentAuthor:
   for i in range(len(df_cleaned)):
      if df_cleaned.iloc[i]['authId_enc'] == auth:
         val_id.append(i)
         break
train_id = [i for i in df_cleaned.index if i not in val_id]  

train_df = df_cleaned.iloc[train_id]
train_df.reset_index(inplace=True, drop = True)
val_df = df_cleaned.iloc[val_id]
val_df.reset_index(inplace=True, drop = True)
# saving data to file
train_df.to_pickle("data/processed/train_clean_df.pkl")
val_df.to_pickle("data/processed/val_clean_df.pkl")
### Cleaning test data
df_test = pd.read_pickle("data/processed/test_dirty_df.pkl")
df_test['title'] = df_test['title'].apply(process_row)
df_test['abstract'] = df_test['abstract'].apply(process_row)
mergingtext(df_test)
# write back to processed folder
df_test.to_pickle("data/processed/test_clean_df.pkl")



# Stage 1: Build preliminary model with Gridsearch
   # prepare data for Gridsearch: remove authors that have less than 3 papers
      
count = Counter(df_cleaned['authId_enc'])
frequentAuthor = list({k:v for k,v in count.items() if count[k] >=3}.keys())
len(frequentAuthor)
df_3papers = df_cleaned[df_cleaned['authId_enc'].isin(frequentAuthor)].copy()
df_3papers.reset_index(inplace=True, drop = True)

def get_author_unique_corpus(df):
   listauthor = df['authId_enc'].unique()
   corpus_auth = {}
   for au in listauthor:
      totalword = []
      auth_content = df.loc[df['authId_enc'] == au, 'content']
      for art in auth_content:
         wordlist = art.split()
         totalword += wordlist
      corpus_auth[au] = list(set(totalword))
   return corpus_auth
corpus_by_author = get_author_unique_corpus(df_3papers)

num_corpus = {k:len(v) for k, v in corpus_by_author.items() }
max_uniq_w = max(num_corpus, key = num_corpus.get)
print('Checking unique words in author\'s data for Gridsearch:')
print(f'author {max_uniq_w} has maximum unique words: {num_corpus[max_uniq_w]}' )
min_uniq_w = min(num_corpus, key = num_corpus.get)
print(f'author {min_uniq_w} has maximum unique words: {num_corpus[min_uniq_w]}' )
print(f'average number of unique words per author is {np.mean(list(num_corpus.values()))}' )

      # Implementing Gridsearch

Xgrid = df_3papers['content']
ygrid = df_3papers['authId_enc'].values
vectorizer = TfidfVectorizer()
classifierNB = naive_bayes.MultinomialNB()
modelNB = pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])
from sklearn.model_selection import GridSearchCV
grid_params = {
  'classifier__alpha': np.linspace(0.5, 1.5, 3),
  'classifier__fit_prior': [True, False],
  'vectorizer__max_df': [0.1, 0.5, 1],
  'vectorizer__binary': [True, False],
  'vectorizer__norm': [None, 'l1', 'l2'], 
  'vectorizer__max_features': [1000, 1500, 2000, 2500, 3000]
}
clf = GridSearchCV(modelNB, grid_params, cv = 3, scoring='accuracy')
clf.fit(Xgrid, ygrid)


print('Best parameter')
clf.best_params_
clf.cv_results_
clf.best_score_
np.mean(clf.cv_results_['mean_test_score'])
min(clf.cv_results_['mean_test_score'])
max(clf.cv_results_['mean_test_score'])

# Stage 2: Fine-tuning final model
df_train = pd.read_pickle('data/processed/train_clean_df.pkl')
df_val = pd.read_pickle('data/processed/val_clean_df.pkl')

corpus_by_author = get_author_unique_corpus(df_train)

num_corpus = {k:len(v) for k, v in corpus_by_author.items() }
max_uniq_w = max(num_corpus, key = num_corpus.get)
print('Checking unique words in author\'s corpus in splitted train data')
print(f'author {max_uniq_w} has maximum unique words: {num_corpus[max_uniq_w]}' )
min_uniq_w = min(num_corpus, key = num_corpus.get)
print(f'author {min_uniq_w} has maximum unique words: {num_corpus[min_uniq_w]}' )
print(f'average number of unique words per author is {np.mean(list(num_corpus.values()))}' )

# trial with 3000-word corpus
Xtrain = df_train['content'].values
ytrain = df_train['authId_enc'].values
Xval = df_val['content'].values
yval= df_val['authId_enc'].values

vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=3000,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_3000= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])

modelNB_3000.fit(Xtrain, ytrain)

predicted = modelNB_3000.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 3000-word corpus', accuracy)
##### Increasing the size of corpus
# 3500 words

vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=3500,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_3500= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])

modelNB_3500.fit(Xtrain, ytrain)
predicted = modelNB_3500.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 3500-word corpus', accuracy)

# 4000 words
vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=4000,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_4000= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])

modelNB_4000.fit(Xtrain, ytrain)
predicted = modelNB_4000.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 4000-word corpus', accuracy)

#4500 words
vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=4500,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_4500= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])

modelNB_4500.fit(Xtrain, ytrain)
predicted = modelNB_4500.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 4500-word corpus', accuracy)

# 5000 words
vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=5000,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_5000= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])
modelNB_5000.fit(Xtrain, ytrain)
predicted = modelNB_5000.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 5000-word corpus', accuracy)

#6000 words
Xtrain = df_train['content']
ytrain = df_train['authId_enc'].values
vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=6000,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_6000= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])

modelNB_6000.fit(Xtrain, ytrain)
predicted = modelNB_6000.predict(Xval)
accuracy = accuracy_score(yval, predicted)
print('accuracy in 6000-word corpus', accuracy)


#### STACKING DATA FOR REFIT CHOSEN MODEL:3500 words
df_combined = pd.concat([df_train, df_val], ignore_index=True, axis=0)
Xcombined = df_combined['content'].values
ycombined = df_combined['authId_enc'].values

vectorizer = TfidfVectorizer(max_df=0.5,
                              max_features=3500,
                              norm=None)
classifierNB = naive_bayes.MultinomialNB(fit_prior=False,alpha=0.5)
modelNB_3500= pipeline.Pipeline([("vectorizer", vectorizer),  
                           ("classifier", classifierNB)])
modelNB_3500.fit(Xcombined, ycombined)
df_test = pd.read_pickle('data/processed/test_clean_df.pkl')

# get lable encoder of author
with open('code/authorIdlabel.pkl', 'rb') as f:
   authorId_encoder = pickle.load(file = f)

Xtest = df_test['content'].values
predicted = modelNB_3500.predict(Xtest)
predictauthorId = authorId_encoder.inverse_transform(predicted)

predictauthorId = list(map(str, predictauthorId))
df_test['authorId']= predictauthorId
df_test = df_test[['paperId', 'authorId']].copy()
df_test.to_json('data/processed/predicted_full.json', orient="records")