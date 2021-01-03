from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

def show_performance_of_model(model, test_data, test_labels):
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

def train_decision_tree(train_data, train_labels, test_data, test_labels):
    #features = train_data.columns.values
    model = DecisionTreeClassifier()
    model.fit(train_data, train_labels)

    show_performance_of_model(model, test_data, test_labels)

    #predictions = model.predict(test_data)


    #importances = model.feature_importances_
    # Sort the importance
    #indices = np.argsort(importances)[::-1]
    
    # Reorder feature names so that they match the sorted feature importance
    #names = [features[i] for i in indices]
    
    #print(names)

    #plt.figure(figsize=(100,10))
    #plt.title("Importance of the features")
    #plt.bar(range(train_data.shape[1]), importances[indices])
    #plt.xticks(range(train_data.shape[1]), names, rotation=90)
    #plt.grid(axis='y')
    #fig1 = plt.gcf()
    #plt.show()
    #plt.draw()
    #fig1.savefig('Features_Importance_DecisionTree.png', dpi=300)

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

    return model, selector

def kfold_decision_tree(train_data, train_labels, test_data, test_labels, splits = 3):
    model = DecisionTreeClassifier()
    model.fit(train_data, train_labels)
    
    # Create k-fold cross-validation for decision tree
    kf = KFold(n_splits=splits, shuffle=False, random_state=1)
    # Perform k-fold cross-validation
    validation_results = cross_val_score(model,test_data, test_labels, cv=kf, scoring="accuracy",n_jobs=-1) 
    
    #Validation Results
    print("Validation results")
    print(validation_results)
    
    # Mean of validation
    print("Mean accuracy")
    print(validation_results.mean())

    return model

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

    return names, importances

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

    return model, selector

def kfold_random_forest(train_data, train_labels, test_data, test_labels, splits = 3):
    randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
    model = randomforest.fit(train_data, train_labels)
    
    # Create k-fold cross-validation for decision tree
    kf = KFold(n_splits=splits, shuffle=False, random_state=1)
    # Perform k-fold cross-validation
    validation_results = cross_val_score(model,test_data, test_labels, cv=kf, scoring="accuracy",n_jobs=-1) 
    
    #Validation Results
    print("Validation results")
    print(validation_results)
    
    # Mean of validation
    print("Mean accuracy")
    print(validation_results.mean())

    return model


def visualize_nn_result(history):

    # Die Verläufe von Trainings- und Testverlusten abrufen
    training_loss = history.history["loss"]
    test_loss = history.history["val_loss"]
    # Zähler für die Anzahl der Epochen erstellen
    epoch_count = range(1, len(training_loss) + 1)

    # Verlustverlauf visualisieren
    plt.plot(epoch_count, training_loss, "r--")
    plt.plot(epoch_count, test_loss, "b-")
    plt.legend(["Trainingsverlust", "Testverlust"])
    plt.xlabel("Epoche")
    plt.ylabel("Verlust")
    plt.show()

    # Verläufe von Trainings- und Testgenauigkeit abrufen
    training_accuracy = history.history["accuracy"]
    test_accuracy = history.history["val_accuracy"]
    plt.plot(epoch_count, training_accuracy, "r--")
    plt.plot(epoch_count, test_accuracy, "b-")
    # Genauigkeitsverlauf visualisieren
    plt.legend(["Trainingsgenauigkeit", "Testgenauigkeit"])
    plt.xlabel("Epoche")
    plt.ylabel("Genauigkeit")
    fig1 = plt.gcf()
    plt.show()

def create_neural_network_1_node(optimizer='adam', activation="sigmoid", neurons=100):

    network = models.Sequential()

    network.add(layers.Dense(units=250,activation="relu",input_shape=(1,)))
    network.add(layers.Dense(units=4, activation="softmax"))

    network.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return network
 
def grid_search_neural_network(data, labels):
    np.random.seed(0)
    data_train = np.asmatrix(data)
    target_train = to_categorical(labels.to_numpy()-1, num_classes=4)
    neural_network = KerasClassifier(build_fn=create_neural_network_1_node, verbose=0)

    epochs = [5, 10, 20]
    batches = [5, 100]
    optimizers = ["rmsprop", "adam"]
    activation = ["relu", "sigmoid"]
    neurons = [100, 500]

    hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, activation=activation, neurons=neurons)

    # Gittersuche erstellen
    grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters, n_jobs=-1)
    # Gittersuche anpassen
    grid_result = grid.fit(data_train, target_train)

    print( grid_result.best_params_)

def neural_network_1_node(train_data, train_labels, test_data, test_labels, no_features):
    
    np.random.seed(0)
    number_of_features = no_features

    tokenizer = Tokenizer(num_words=number_of_features)
    features_train = np.asmatrix(train_data)
    features_test = np.asmatrix(test_data)

    target_train = to_categorical(train_labels.to_numpy()-1, num_classes=4)
    target_test = to_categorical(test_labels.to_numpy()-1, num_classes=4)

    network = create_neuronal_network_1_node()

    #print(features_train.shape)
    #print(target_train.shape)
    #print(target_train)

    # Callback-Funktionen einrichten, um Training frühzeitig zu stoppen und das
    # bislang beste Modell zu speichern
    #callbacks = [EarlyStopping(monitor="val_loss", patience=50),
    #ModelCheckpoint(filepath="best_model.h5",
    #monitor="val_loss",
    #save_best_only=True)]

    # Neuronales Netz trainieren
    history = network.fit(features_train, # Merkmale
    target_train, # Ziel
    epochs=1000,
    #callbacks=callbacks,
    verbose=0, # Keine Ausgabe
    batch_size=30, # Anzahl der Beobachtungen pro Batch
    validation_data=(features_test, target_test)) # Testdaten

    visualize_nn_result(history)

