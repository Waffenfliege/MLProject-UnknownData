from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt


def train_decision_tree(train_data, train_labels, test_data, test_labels):
    #features = train_data.columns.values
    model = DecisionTreeClassifier()
    model.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    print("### Results on test set: ###")

    acc = accuracy_score(test_labels, predictions)
    print("Overall accuracy: ", acc)

    print("Confusion matrix")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print("Classification Report")
    cr = classification_report(test_labels, predictions)
    print(cr)

    importances = model.feature_importances_
    # Sort the importance
    indices = np.argsort(importances)[::-1]
    
    # Reorder feature names so that they match the sorted feature importance
    names = [features[i] for i in indices]
    
    print(names)

    plt.figure(figsize=(100,10))
    plt.title("Importance of the features")
    plt.bar(range(train_data.shape[1]), importances[indices])
    plt.xticks(range(train_data.shape[1]), names, rotation=90)
    plt.grid(axis='y')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('Features_Importance_DecisionTree.png', dpi=300)

def train_decison_tree_only_important_features(train_data, train_labels, test_data, test_labels, threshold):
    
    decisionTree = DecisionTreeClassifier(random_state=0)
    selector = SelectFromModel(decisionTree, threshold=threshold)
    
    features_important = selector.fit_transform(train_data, train_labels)
    print(features_important.shape)
    
    model = decisionTree.fit(features_important, train_labels)

    predictions = model.predict(selector.transform(test_data))
    
    acc = accuracy_score(test_labels, predictions)
    print("Overall accuracy: ", acc)

    print("Confusion matrix")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print("Classification Report")
    cr = classification_report(test_labels, predictions)
    print(cr)

def kfold_decision_tree(train_data, train_labels, test_data, test_labels, splits = 3):
    model = DecisionTreeClassifier()
    model.fit(train_data, train_labels)
    
    # Create k-fold cross-validation for decision tree
    kf = KFold(n_splits=splits, shuffle=True, random_state=1)
    # Perform k-fold cross-validation
    validation_results = cross_val_score(model,test_data, test_labels, cv=kf, scoring="accuracy",n_jobs=-1) 
    
    #Validation Results
    print("Validation results")
    print(validation_results)
    
    # Mean of validation
    print("Mean accuracy")
    print(validation_results.mean())

def train_random_forest(train_data, train_labels, test_data, test_labels):
    
    randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
    model = randomforest.fit(train_data, train_labels)

    predictions = model.predict(test_data)
    
    acc = accuracy_score(test_labels, predictions)
    print("Overall accuracy: ", acc)

    print("Confusion matrix")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print("Classification Report")
    cr = classification_report(test_labels, predictions)
    print(cr)

def importance_of_features_random_forest(train_data, train_labels, test_data, test_labels):
    features = train_data.columns.values

    randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
    model = randomforest.fit(train_data, train_labels)

    importances = model.feature_importances_
    # Sort the importance
    indices = np.argsort(importances)[::-1]
    
    # Reorder feature names so that they match the sorted feature importance
    names = [features[i] for i in indices]
    
    print(names)

    plt.figure(figsize=(100,10))
    plt.title("Importance of the features")
    plt.bar(range(train_data.shape[1]), importances[indices])
    plt.xticks(range(train_data.shape[1]), names, rotation=90)
    plt.grid(axis='y')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('Features_Importance_RandomForest.png', dpi=300)

def train_random_forest_only_important_features(train_data, train_labels, test_data, test_labels, threshold):
    
    randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)

    selector = SelectFromModel(randomforest, threshold=threshold)
    

    features_important = selector.fit_transform(train_data, train_labels)
    print(features_important.shape)

    model = randomforest.fit(features_important, train_labels)

    predictions = model.predict(selector.transform(test_data))
    
    acc = accuracy_score(test_labels, predictions)
    print("Overall accuracy: ", acc)

    print("Confusion matrix")
    cm = confusion_matrix(test_labels, predictions)
    print(cm)

    print("Classification Report")
    cr = classification_report(test_labels, predictions)
    print(cr)

def kfold_random_forest(train_data, train_labels, test_data, test_labels, splits = 3):
    randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
    model = randomforest.fit(train_data, train_labels)
    
    # Create k-fold cross-validation for decision tree
    kf = KFold(n_splits=splits, shuffle=True, random_state=1)
    # Perform k-fold cross-validation
    validation_results = cross_val_score(model,test_data, test_labels, cv=kf, scoring="accuracy",n_jobs=-1) 
    
    #Validation Results
    print("Validation results")
    print(validation_results)
    
    # Mean of validation
    print("Mean accuracy")
    print(validation_results.mean())