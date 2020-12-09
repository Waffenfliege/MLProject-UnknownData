import pandas as pd
import matplotlib.pyplot as plt

from random import randint


class Data:

    def __init__(self, file):
        self.file = file
        self.raw_data = self.read_dataset()

    def printData(self):
        print(self.file)

    #Read data from csv
    def read_dataset(self):
        data = pd.read_csv(self.file)
        return(data)
    
    #Get the data, its features and labels
    def get_data_features_labels(self):
        data = self.raw_data.copy()
        
        #Get the data labels/classes
        labels = data['class']

        #Delete feature - class 
        data.pop('class')

        #Get the data features
        features = data.columns.values

        return(data, features, labels)
    
    #Create random blind test set in 30 chunks and drop it from train test set
    def create_blind_test_set(self, no_of_datarecords=3, path="../data/processed/"):
        reduced_dataset = self.raw_data.copy()
        blind_dataset = self.raw_data[0:0]

        selected_data_records = []
        no_data_records = len(self.raw_data)/30

        for i in range(no_of_datarecords):
            random_no = randint(1,no_data_records)
            if random_no not in selected_data_records:
                selected_data_records.append(random_no)
        
        selected_data_records= sorted(selected_data_records, reverse=True)

        for i in selected_data_records:
            no_row = (i-1)*30
            
            blind_dataset = blind_dataset.append(self.raw_data.iloc[no_row:(no_row+30)])
            reduced_dataset = reduced_dataset.drop(reduced_dataset.index[no_row:(no_row+30)])
        
        blind_dataset.to_csv(path_or_buf=path+"blind_dataset.csv", index=False)
        reduced_dataset.to_csv(path_or_buf=path+"train_dataset.csv", index=False)

        return(reduced_dataset, blind_dataset)

    def get_min_max_mean(self):

        #Check if there are empty entries
        emtpy_values = self.raw_data.isnull().any().sum()

        #Get the min and max values of each features
        min_values = self.raw_data.min()
        max_values = self.raw_data.max()
        mean_values = self.raw_data.mean()

        #Get the min and max of the complete dataset
        min_value = min_values.min()
        max_value = max_values.max()
        mean_value = mean_values.mean()

        print('No. of empty values: ' + str(emtpy_values))
        print('Max value of Dataset: ' + max_values.idxmax() +' '+ str(max_value)) #=> DH69 Highest Value of Dataset
        print('Min value of Dataset: ' + min_values.idxmin() +' '+ str(min_value)) #=> T15 lowest Value of Dataset
        print('Mean value of Dataset: ' + str(mean_value))

    def plot_mean_values(self):
        self.raw_data.mean().plot()
        plt.title("Mean value of each feature")
        plt.show()

    def plot_min_values(self):
        self.raw_data.min().plot()
        plt.title("Minimum value of each feature")
        plt.show()

    def plot_max_values(self):
        self.raw_data.max().plot()
        plt.title("Maximum value of each feature")
        plt.show()

    def plot_number_entries_for_class(self):
        #Visualize a number of entries of the respective class
        data, features, labels = self.get_data_features_labels()
        count_labels = labels.value_counts().sort_index()
        print(count_labels)
        count_labels.plot(kind = 'bar', rot=0)
        plt.title("Distribution of the entries to the classes")
        plt.xticks(range(4), count_labels.index)
        plt.xlabel("Class")
        plt.ylabel("Entries")
        plt.show()
        