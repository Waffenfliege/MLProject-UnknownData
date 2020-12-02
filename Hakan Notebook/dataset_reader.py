import pandas as pd

def read_dataset_raw():
    #Read data from csv
    data = pd.read_csv("train.csv")
    return(data)

def read_dataset():
    data = read_dataset_raw()
    
    #Get the data labels/classes
    labels = data['class']

    #Delete feature - class 
    data.pop('class')

    #Get the data features
    features = data.columns.values

    return(data, features, labels)
