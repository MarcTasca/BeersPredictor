# ---------- IMPORTS ----------

import string
import re
import pandas as pd
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from matplotlib.legend_handler import HandlerLine2D

# ---------- USEFUL FUNCTIONS ----------

# write the csv to submit on the platform
def write_submission(x, y, name):
  result = pd.DataFrame()
  result['Id'] = x.index
  result['Predicted'] = y
  result.to_csv(os.path.join(data, name), index=False, header=True, sep=',')

# stemming every words from dataset's reviews
def stemming(review):
  words = word_tokenize(review)
  ps = PorterStemmer()
  return " ".join([ps.stem(w) for w in words])

# preprocess the dataset's reviews, except for stemmization
def preprocessing_text(input_str):
  # removing numbers
  input_str = re.sub(r'\d+', '', input_str)
  # to lower
  input_str = input_str.lower()
  # removing puntuaction
  input_str = input_str.translate(str.maketrans('', '', string.punctuation))
  # remove white spaces
  input_str = " ".join(input_str.split())
  # stemming the review
  # input_str = stemming(input_str)
  return input_str

# fasttext unsupervised training -> my_little_mushroom.bin
# it takes 16 minutes straight
def train_fasttext(df):
  feed = open(r"food_for_fasttext.txt","w")
  feed.write(''.join(df['review/text'].values))
  feed.close()
  model = fasttext.train_unsupervised('food_for_fasttext.txt', dim=150)
  model.save_model('/content/drive/MyDrive/progetto dsl/fasttext_model/my_little_mushroom.bin')
  return model, 'my_little_mushroom.bin'

# fasttext sentence vectorizer on dataset's reviews
def text_transformation_fasttext(df, model):
  ft = df['review/text'].apply(model.get_sentence_vector).apply(pd.Series)
  df_ft = pd.concat([df, ft], axis=1)
  return df_ft

# countvectorizer + svd
# it could also be tfidf
# you just need to add .toarray() to the matrix and to comment and uncomment two lines of code
def text_transform(train, test, mode):
  #model = TfidfVectorizer(max_df=0.9, min_df=0.1)
  model1 = CountVectorizer()
  model2 = TruncatedSVD(n_components=150)
  model = Pipeline([('countvect',model1),('svd',model2)])
  # train
  train_matrix = model.fit_transform(train['review/text'])
  train_ft = pd.concat([train, pd.DataFrame(train_matrix, index=train.index)], axis=1)

  # test
  test_matrix = model.transform(test['review/text'])
  test_ft = pd.concat([test, pd.DataFrame(test_matrix, index=test.index)], axis=1)

  return train_ft, test_ft

# one hot encoder for beer_style return train and test
def ohe_beer(train_ft, test_ft, mode):  
  model = OneHotEncoder(handle_unknown='ignore')
  #model2 = TruncatedSVD(n_components=15)
  #model = Pipeline([('onehot',model1),('pca',model2)])

  # train
  beer_style = train_ft[['beer/style']]
  encoded_matrix = model.fit_transform(beer_style)
  train_ft_b = pd.concat([train_ft, pd.DataFrame(encoded_matrix.toarray(), index=beer_style.index)], axis=1)

  # test
  beer_style = test_ft[['beer/style']]
  encoded_matrix = model.transform(beer_style)
  test_ft_b = pd.concat([test_ft, pd.DataFrame(encoded_matrix.toarray(), index=beer_style.index)], axis=1)

  return train_ft_b, test_ft_b

# drop unused columns prepare and return x and y
def drop_prepare(train_ft_b, test_ft_b, mode):
  x_train = train_ft_b.drop(columns=['review/overall','review/text','beer/style'])
  if mode == 'development':
    x_test = test_ft_b.drop(columns=['review/overall','review/text','beer/style'])
  elif mode == 'evaluation':
    x_test = test_ft_b.drop(columns=['review/text','beer/style'])

  y_train = train_ft_b['review/overall']
  if mode == 'development':
    y_test = test_ft_b['review/overall']
  else: y_test = None

  return x_train, y_train, x_test, y_test

# return the score of the selected model
def score_model(x_train, y_train, x_test, y_test, model):
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  
  return r2_score(y_pred,y_test)

# train the model, return the prediction and groung truth
def train_model(x_train, y_train, x_test, model):
  model.fit(x_train,y_train)
  y_pred = model.predict(x_test)
  return x_test, y_pred

# tuning with graphs the hyperparameters of rigde with polynomial_features
def model_score(x_train, y_train, x_test, y_test, models):
  # r2_score results
  train_results = []
  test_results = []

  for model in models:
    print('-', end='')
    # train the model
    model.fit(x_train, y_train)

    # r2_score for training set
    train_pred = model.predict(x_train)
    out = r2_score(y_train, train_pred)
    train_results.append(out)

    # r2_score for test set
    y_pred = model.predict(x_test)
    out = r2_score(y_test, y_pred)
    test_results.append(out)

  return train_results, test_results

# plot the results
def plot_r2score(tuning_param, xlabel, train_results, test_results):
  line1, = plt.plot(tuning_param, train_results, 'b', label='Train R2')
  line2, = plt.plot(tuning_param, test_results, 'r', label='Test R2')
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
  plt.ylabel('R2 score')
  plt.xlabel(xlabel)
  plt.show()

# manual cross validation returning r2_score
def manual_cross_validation(df, n, model, mode):
  kfold = KFold(n_splits=n, random_state=21, shuffle=True)
  score = 0

  for train_id, test_id in kfold.split(df):
    # counting fold
    # print('| ',end='')

    # split train and test
    train_ft = df.iloc[train_id]
    test_ft = df.iloc[test_id]

    # train_ft, test_ft = text_transform(train, test, mode)
    train_ft_b, test_ft_b = ohe_beer(train_ft, test_ft, mode)
    x_train, y_train, x_test, y_test = drop_prepare(train_ft_b, test_ft_b, mode)

    score += score_model(x_train, y_train, x_test, y_test, model)
  return score/n
