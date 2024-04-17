#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:13:29 2024

@author: saradelasota
"""


import pandas as pd
import time as t

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from evaluation import append_evaluation_metrics_to_csv

class KNNModel:
    def __init__(self, n_neighbors, weights='uniform', algorithm='auto',p=2):
        '''
        Initialize the KNN model
        :param n_neighbors: Number of neighbors to use
        :param weights: Weight function used in prediction
        :param algorithm: Algorithm used to compute the nearest neighbors
        :param p: Power parameter for the Minkowski metric. When p = 1 it is equivalent to the manhattan_distance
                  when p = 2 it is equivalent to the euclidean_distance
        '''
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.p = p
        self.model = None

    def fit(self, X_train, y_train):
        '''
        Fit the KNN model to the training data
        :param X_train: The training features
        :param y_train: The training labels
        '''
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                          weights=self.weights,
                                          algorithm=self.algorithm, p = self.p)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        '''
        Make predictions using the trained KNN model
        :param X_test: The test features
        :return: Predicted labels
        '''
        if self.model is None:
            raise ValueError("KNN model has not been trained yet. Call fit() first.")
        return self.model.predict(X_test)
    

def main():
    df = pd.read_csv('datasets/df_selection2.csv')
    dataset_name = 'df_selection2.csv'
    # Split the data into features (X) and target variable (y)
    X = df.drop('url', axis=1)
    X = X.drop('status', axis=1)  # Features
    y = df['status']  # Target variable
    
    # Random state
    rs = 123
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
   
    # Define the parameter grid for KNN
    param_grid = {
        'n_neighbors': [2,5,10,15,20,30],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'p': [1, 2]  # Values for the power parameter for the Minkowski metric
    }
    
    # Create a K-nearest neighbors classifier
    knn_classifier = KNeighborsClassifier()
    
    # Instantiate GridSearchCV for KNN
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("Best parameters found for KNN:", best_params)
    print("Best accuracy score for KNN:", best_score)
    
    # Instantiate the KNNModel object with desired parameters
    knn_model = KNNModel(**best_params)
    
    start_time = t.time()

    # Fit the model to the training data
    knn_model.fit(X_train, y_train)
    
    # Calculate time taken
    end_time = t.time()
    fit_time = end_time - start_time
    
    start_time = t.time()
    # Make predictions on the test data
    predictions = knn_model.predict(X_test)
    
    # Calculate time taken
    end_time = t.time()
    predict_time = end_time - start_time
    
   
    # Evaluate the model
    model_name = 'KNeighborsClassifier'
    knn_accuracy = accuracy_score(y_test, predictions)
    knn_precision = precision_score(y_test, predictions)
    knn_recall = recall_score(y_test, predictions)
    knn_f1 = f1_score(y_test, predictions)
    
    print("Test accuracy:", knn_accuracy)
    print("Test precision:", knn_precision)
    print("Test recall:", knn_recall)
    print("Test f1 score:", knn_f1)
        
    evaluation_metrics = {
    'accuracy': knn_accuracy,
    'precision': knn_precision,
    'recall': knn_recall,
    'f1_score': knn_f1
    }
    

    # Call the function to append the new evaluation metrics to the existing CSV file
    append_evaluation_metrics_to_csv(model_name, evaluation_metrics,dataset_name,best_params, fit_time,predict_time)
    
    
if __name__ == "__main__":
    main()
