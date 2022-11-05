import base64
import matplotlib as mt
mt.use('Agg')
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
import io

def preprocessing(csv_file):
    ## Preprocessing of the dataset
    df = pd.read_csv(csv_file)
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
    kn_model = KNeighborsClassifier(n_neighbors=15)
    kn_model.fit(X_train,y_train)
    kn_score = kn_model.score(X_test,y_test)
    models_accuracy['KNN'] = kn_score*100
    ## Random Forest Classifier Model
    ran_model = RandomForestClassifier(n_estimators=40)
    ran_model.fit(X_train,y_train)
    ran_score = ran_model.score(X_test,y_test)
    models_accuracy['RandomForest'] = ran_score*100
    ## AdaBoost Classifier
    ada_model = AdaBoostClassifier()
    ada_model.fit(X_train,y_train)
    ada_score = ada_model.score(X_test,y_test)
    models_accuracy['AdaBoost'] = ada_score*100
    ## Gradient Boost Classifier 
    grad_model = GradientBoostingClassifier()
    grad_model.fit(X_train,y_train)
    grad_score = grad_model.score(X_test,y_test)
    models_accuracy['Gradient'] = grad_score*100
    pred = ran_model.predict(X_test)
    ## Printing Classification report 
    classification_reports = classification_report(y_test,pred, output_dict=True)
    ## Printing Confusion Matrix
    plot_confusion_matrix(ran_model,
                     X_test,y_test,
                     display_labels=['Placed','Not Placed'])
    plt.grid(False)
    fig = io.BytesIO()
    plt.savefig(fig, format='png')
    plt.close()
    
    return {
        "model_accuracy" : models_accuracy,
        "classification_report" : classification_reports,
        "confusion_matrix" : base64.b64encode(fig.getvalue()).decode(),
    }

