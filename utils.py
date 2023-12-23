## Main Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
import missingno
warnings.filterwarnings('ignore')
## sklearn -- preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
#import GridSearchCV
from sklearn.model_selection import GridSearchCV

## sklearn -- models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score



## skelarn -- metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



df=pd.read_csv(r'C:\Users\elsay\Desktop\Internship project\zomato.csv')

dfa=df.drop(['url','phone','dish_liked','reviews_list','menu_item','address'],axis=1)

dfa.drop_duplicates(inplace=True)
dfa.dropna(inplace=True)
dfa.rename(columns={'approx_cost(for two people)':'cost'},inplace=True)
dfa.rename(columns={'listed_in(type)':'type'},inplace=True)
dfa.rename(columns={'listed_in(city)':'city'},inplace=True)

dfa['cost']=dfa['cost'].astype(str)
dfa['cost']=dfa['cost'].apply(lambda x:x.replace(',','.'))
dfa['cost']=dfa['cost'].astype(float)
dfa=dfa.loc[dfa.rate !='NEW'] ## Removing 'New'
dfa=dfa.loc[dfa.rate !='-'].reset_index(drop=True) ## Removing '-'

def remove_slash(x):
    if isinstance(x, str):
        return x.replace('/5', '').strip()
    return x

# Apply the function to the "rate" column
dfa.rate = dfa.rate.apply(remove_slash)

# Convert the "rate" column to a numeric type
dfa['rate'] = pd.to_numeric(dfa['rate'], errors='coerce')

dfa['rate'] = dfa['rate'].apply(lambda x: round(x, 1) if x % 1 == 0 else x)

dfa.name = dfa.name.apply(lambda x:x.title())
dfa.online_order.replace(('Yes','No'),(True, False),inplace=True)
dfa.book_table.replace(('Yes','No'),(True, False),inplace=True)

threshold_rate = 3.75


dfa['is_good'] = (dfa['rate'] >= threshold_rate).astype(int)

dfaa = dfa.drop('rate', axis=1)

## to features and target
X = dfaa.drop(columns=['is_good'], axis=1)
y = dfaa['is_good']


## split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)

num_cols=['votes','cost']
categ_cols=['name','online_order','book_table','location','rest_type','cuisines','type','city']

num_pipline = Pipeline(steps=[
                ('selector', DataFrameSelector(num_cols)),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

## Categorical
categ_pipline = Pipeline(steps=[
                 ('selector', DataFrameSelector(categ_cols)),
                 ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False,handle_unknown='ignore'))
])

all_pipeline = FeatureUnion(transformer_list=[
                        ('num', num_pipline),
                        ('categ', categ_pipline)
                    ])

## apply
_= all_pipeline.fit_transform(X_train)


def process_new(x_new):
    df_new=pd.DataFrame([x_new],columns=X_train.columns)
    

    ##Adjust the datatypes
    df_new['name']=df_new['name'].astype('object')
    df_new['online_order']=df_new['online_order'].astype('int64')
    df_new['book_table']=df_new['book_table'].astype('int64')
    df_new['votes']=df_new['votes'].astype('int64')
    df_new['location']=df_new['location'].astype('object')
    df_new['rest_type']=df_new['rest_type'].astype('object')
    df_new['cuisines']=df_new['cuisines'].astype('object')
    df_new['cost']=df_new['cost'].astype('float64')
    df_new['type']=df_new['type'].astype('object')
    df_new['city']=df_new['city'].astype('object')
    
   

    ## Apply the pipeline
    X_processed=all_pipeline.transform(df_new)


    return X_processed

