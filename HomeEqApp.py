import streamlit as st
import pandas as pd

#tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

#pipeline tools
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#ml related
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


#image

from PIL import Image
image = Image.open('Data/House.png')
st.image(image)


#variables
Continuous = ["MORTDUE", "VALUE", "CLAGE", "DEBTINC"]
Discrete = ["YOJ", "DEROG", "DELINQ", "NINQ", "CLNO"]
Cats = ['REASON','JOB']

#title
st.write("""
# Home Equity Loan Prediction App
This app predicts if the applicant will **return the loan or not**!!
""")
st.write('---')

# Loads the Data For training
df1 = pd.read_csv("Data/hmeq.csv")
X = df1.drop("BAD", axis = 1)
y = df1["BAD"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify The Clients Parameters')
def user_input_features():
    LOAN = st.sidebar.slider('Loan Amount', 1100, 89800)
    MORTDUE = st.sidebar.slider('Amount due on existing mortgage', 73807, 399550)
    VALUE = st.sidebar.slider('Value of property', 8800, 855909)
    REASON = st.sidebar.selectbox('Reason for Loan', options = ['HomeImp', 'DebtCon'])
    JOB = st.sidebar.selectbox('Job', options = ['Mgr', 'Office', 'Other', 'ProfExe', 'Self', 'Sales'])
    YOJ = st.sidebar.slider('Years in current job', 0, 41, step = 1)
    DEROG = st.sidebar.slider('Number of major derogatory reports', 0, 10, step = 1)
    DELINQ = st.sidebar.slider('Number of delinquent credit lines', 0, 12, step = 1)
    CLAGE = st.sidebar.slider('Age of oldest trade line in months',0, 1169)
    NINQ = st.sidebar.slider('Number of recent credit lines', 0, 17, step = 1)
    CLNO = st.sidebar.slider('Number of credit lines', 0, 71, step = 1)
    DEBTINC = st.sidebar.slider('Debt-to-income ratio', 0.7, 144.2)
    data = {'LOAN': LOAN,
            'MORTDUE': MORTDUE,
            'VALUE': VALUE,
            'REASON': REASON,
            'JOB': JOB,
            'YOJ': YOJ,
            'DEROG': DEROG,
            'DELINQ': DELINQ,
            'CLAGE': CLAGE,
            'NINQ': NINQ,
            'CLNO': CLNO,
            'DEBTINC': DEBTINC}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

#pre processor
# Preprocessing for continuous data use best settings from the previus baseline preprocessor grid search
contiuous_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='mean')),
('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
('imputer', SimpleImputer(strategy='constant')),
('oh', OneHotEncoder())
])

# Preprocessing for discrete data
disc_transformer = Pipeline(steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
    ])

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', contiuous_transformer, Continuous+["LOAN"]),
        ('cat', categorical_transformer, Cats),
        ('disc', disc_transformer, Discrete)
    ])

#preprocess
preprocess = Pipeline(steps = [('preprocess', preprocessor)])
X_train = preprocess.fit_transform(X_train)
y_train = y_train.to_numpy()
# define new model with basic parameters and load state dict weights
clf = TabNetClassifier(verbose=0,seed=42)
clf.fit(X_train, y_train)

#preprocess df 
Data = preprocess.transform(df)



#predict probabilities
Predprob = pd.DataFrame(clf.predict_proba(Data),columns = ["Returns Loan", "Default"])
st.header('Probabilities of Home Eq')
st.write(Predprob)
st.write('---')


#prediction
prediction = str(np.where(clf.predict_proba(Data)[:,1]<0.5,"Returns Loan","Default"))[2:-2]
st.header('Prediction of Home Eq')
st.write(prediction)
st.write('---')