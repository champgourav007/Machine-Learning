import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix
import pickle
from sklearn.metrics import accuracy_score
import io
from . import algorithm_test


def preprocessing():
    global y_test    
    ## Preprocessing of the dataset 
    df = pd.read_csv('collegePlace.csv')
    df = pd.get_dummies(df, columns=['Stream'],drop_first=True)
    df = pd.get_dummies(df, columns=['Gender'],drop_first=True)
    
    ## Modelling and training of the data 
    X = df.drop('PlacedOrNot',axis = 1)
    y = df['PlacedOrNot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    ## Models accuracy list with multiple models 
    models_accuracy = {}
    cv = KFold(n_splits=15,random_state=13,shuffle=True)
    
    ## SVC Model 
    svm_model = SVC(decision_function_shape='ovr')
    svm_model.fit(X_train,y_train)
    svm_score = svm_model.score(X_test,y_test)
    models_accuracy['SVM'] = svm_score*100
    
    ## KNneighborsClassifier Model
    global kn_model
    kn_model = KNeighborsClassifier(n_neighbors=15)
    kn_model.fit(X_train,y_train)
    kn_score = kn_model.score(X_test,y_test)
    models_accuracy['KNN'] = kn_score*100
    
    ## Random Forest Classifier Model
    global ran_model
    ran_model = RandomForestClassifier(n_estimators=40)
    ran_model.fit(X_train,y_train)
    ran_score = ran_model.score(X_test,y_test)
    models_accuracy['RandomForest'] = ran_score*100
    
    ## AdaBoost Classifier
    global ada_model
    ada_model = AdaBoostClassifier()
    ada_model.fit(X_train,y_train)
    ada_score = ada_model.score(X_test,y_test)
    models_accuracy['AdaBoost'] = ada_score*100
   
    ## Gradient Boost Classifier
    global grad_model 
    grad_model = GradientBoostingClassifier()
    grad_model.fit(X_train,y_train)
    grad_score = grad_model.score(X_test,y_test)
    models_accuracy['Gradient'] = grad_score*100
    
    ## Saving the best models
    return saving()

    ## Printing Models Accuracy
    # print(models_accuracy)
    # pred = ran_model.predict(X_test)
    

def saving():
    file = io.BytesIO()
    data = pickle.dump(ran_model, file)
    b = bytes(file.getvalue())
    return algorithm_test.testing(b)
