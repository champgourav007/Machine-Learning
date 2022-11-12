import pickle 
from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix
import pandas as pd

def testing(randomForestFile):
    try:
        df = pd.read_csv('form.csv')
        RandomForest = pickle.loads(randomForestFile)
        result = RandomForest.predict(df)
    
        if result[0] == 1:
            return True
        else:
            return False
    except:
        return False