import numpy as np
import pandas as pd
from os import getcwd
import glob
from os.path import join
import ntpath
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle

import warnings
import multiprocessing 
from sklearn.metrics import accuracy_score


def set_path():
   path = 'C:/Users/Eigenaar/1. my_folder/2. Tilburg Data Science/S2B2_Machine learning/ML-classification-Author-identification'
   os.chdir(path)



def score_NLP(auth_NLP,vectorizer,testcase):#all words in train dictionary of one authorb
    auth_nlp = auth_NLP.toarray()
    testTF = vectorizer.transform([testcase]).toarray()[0]
    score = []
    for article in auth_nlp:
        score.append(np.dot(article, testTF))
    return np.mean(score)



def score_Place(placeMod, testcase):
    score = placeMod.predict_proba([testcase])[0][1]
    return score
def calculate_final_score(score_nlp, score_place):
   final_score = score_nlp + score_place 
   return final_score
# prediction for one case:
def prediction(year, venue,content):
   # preparing test cases: 
   testcase_NLP = content
   #testcase_place = [year, venue]

   # access archive:

   pkl_file_list = glob.glob(join(getcwd(),'models_TF',"*.pkl"))
   result_records = {}
   for fname in pkl_file_list:
      _, authid = ntpath.split(fname)
      authid = int(authid[:-4])
      with open(fname, 'rb') as f:
         models = pickle.load(file = f)
      
      # NLP score
      auth_NLP, vectorizer = models[0] 
      score_nlp = score_NLP(auth_NLP, vectorizer,testcase_NLP)

      #place score
      #place_model = models[1]
      #score_place = 0           ############## Remember to change `model_place`` later

      final_score =  score_nlp      ## change later with `calculate_final_score(score_nlp, score_place)``
      result_records[authid] = final_score
   return max(result_records, key= result_records.get)
      

# predict all 
def predict_map(case):
   year = case["year"]
   venue = case["venues_le"]
   content = case["content"]
   pred = prediction(year, venue, content)
   real = case['authId_enc']
   
   return [real, pred]
set_path()
df_val= pd.read_pickle('data/processed/val_clean_df.pkl')

if __name__ == '__main__':
   pool = multiprocessing.Pool(multiprocessing.cpu_count())
   rows = [row for _, row in df_val.iterrows()]
   results = pool.map(predict_map, rows)
   print(results)




def predict_all_parallel():
   set_path()
   df_val= pd.read_pickle('data/processed/val_clean_df.pkl')
   ProPool = multiprocessing.Pool(multiprocessing.cpu_count())
   result = ProPool.map(predict_map, df_val.iterrows())
   ProPool.close()
   return result

warnings.filterwarnings('ignore')
result = predict_all_parallel()

#accuracy_score(df_val['authId_enc'], val_pred)
pred_table = pd.DataFrame(result, columns=['index','real','pred'])
accuracy_score(pred_table['real'], pred_table['pred'])