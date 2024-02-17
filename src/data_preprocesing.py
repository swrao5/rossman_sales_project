import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_engineering import build_features


def load_data(raw_data_path):

   types = {'CompetitionOpenSinceYear': np.dtype(int),
         'CompetitionOpenSinceMonth': np.dtype(int),
         'StateHoliday': np.dtype(str),
         'Promo2SinceWeek': np.dtype(int),
         'SchoolHoliday': np.dtype(float),
         'PromoInterval': np.dtype(str)}
   
   train = pd.read_csv(os.path.join(raw_data_path,'train.csv'), parse_dates=[2], dtype=types)
   test = pd.read_csv(os.path.join(raw_data_path,'test.csv'), parse_dates=[3], dtype=types)
   store = pd.read_csv(os.path.join(raw_data_path,'store.csv'))
   return train, test, store

def preprocess_data(train, test, store, processed_data_path):
   
   #assume the store open if not provided
   train.Open.fillna(1, inplace=True)
   test.Open.fillna(1, inplace=True)

   #Consider only open stores for training. Closed stores wont count into the score
   train = train[train["Open"] != 0]
   #Use only Sales bigger then zero. Simplifies calculation of rmspe
   train = train[train["Sales"] > 0]

   #Join with store
   train = pd.merge(train, store, on='Store')
   test = pd.merge(test, store, on='Store')
   
   features = []
   #augment features
   features_train, train_processed =  build_features(features, train)
   features_test, test_processed = build_features([], test)

   train_processed.to_csv(os.path.join(processed_data_path,'train_processed.csv'))
   test_processed.to_csv(os.path.join(processed_data_path,'test_processed.csv'))

   return train_processed, test_processed, features_train, features_test




   