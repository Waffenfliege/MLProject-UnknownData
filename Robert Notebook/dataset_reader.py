import pandas as pd

def read_dataset_raw():
    #Read data from csv
    data = pd.read_csv(r"../train.csv")
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

def read_dataset_with_hidden(hidden_lines = 3):
    data = read_dataset_raw()
    
    pop_lines = hidden_lines*30

    #Split data in two data sets
    hidden_data = data.tail(pop_lines)
    data = data.head(len(data) - pop_lines)

    #Get the data labels/classes
    labels = data['class']
    hidden_labels = hidden_data['class']

    #Delete feature - class 
    data.pop('class')
    hidden_data.pop('class')

    #Get the data features
    features = data.columns.values

    return(data, features, labels, hidden_data, hidden_labels)