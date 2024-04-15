#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:21:17 2024

@author: saradelasota
"""

import pandas as pd
import time as t

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


from evaluation import append_evaluation_metrics_to_csv


class SupportVectorMachine:
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', rs=123):
        '''
        Initialize the Support Vector Machine (SVM) model
        :param kernel: Specifies the kernel type to be used in the algorithm
        :param C: Regularization parameter
        :param gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        :param rs: Random state for reproducibility
        '''
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.rs = rs
        self.svm = None

    def fit(self, X_train, y_train):
        '''
        Fit the Support Vector Machine (SVM) model to the training data
        :param X_train: The training features
        :param y_train: The training labels
        '''
        self.svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, random_state=self.rs)
        self.svm.fit(X_train, y_train)
        
    def predict(self, X_test):
        '''
        Make predictions using the trained Support Vector Machine (SVM) model
        :param X_test: The test features
        :return: Predicted labels
        '''
        if self.svm is None:
            raise ValueError("SVM model has not been trained yet. Call fit() first.")
        return self.svm.predict(X_test)


def main():

    df = pd.read_csv('datasets/scaled_dataset.csv')
    dataset_name = 'scaled_dataset.csv'
    # Split the data into features (X) and target variable (y)
    X = df.drop('url', axis=1)
    X = X.drop('status', axis=1)  # Features
    y = df['status']  # Target variable
    
    # Random state
    rs = 123
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
    
    # Define the hyperparameters grid to search
    param_grid_svm = {
        'C': [0.1, 1, 10],            # Regularization parameter
        'kernel': ['rbf'],  # Kernel type
        'gamma': [1, 0.1, 0.01,0.001]    # Kernel coefficient for 'rbf'
    }
    
    # Initiate the SVM model
    svm_model = SVC(random_state=rs)

    # Instantiate GridSearchCV
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=5)

    # Fit the grid search to the data
    grid_search_svm.fit(X_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search_svm.best_params_
    best_score = grid_search_svm.best_score_
    
    print("Best parameters found:", best_params)
    print("Best accuracy score:", best_score)
    
    # Instantiate the SVM object with desired parameters
    svm_model = SupportVectorMachine(**best_params, rs=123)
    
    start_time = t.time()

    # Fit the model to the training data
    svm_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions_svm = svm_model.predict(X_test)
    
    # Calculate time taken
    end_time = t.time()
    fit_time = end_time - start_time
    
    # Evaluate the model
    model_name = 'SVM'
    svm_accuracy = accuracy_score(y_test, predictions_svm)
    svm_precision = precision_score(y_test, predictions_svm)
    svm_recall = recall_score(y_test, predictions_svm)
    svm_f1 = f1_score(y_test, predictions_svm)
    
    # Print evaluation metrics
    print("SVM Test accuracy:", svm_accuracy)
    print("SVM Test precision:", svm_precision)
    print("SVM Test recall:", svm_recall)
    print("SVM Test f1 score:", svm_f1)
    
    # Append evaluation metrics to the CSV file
    evaluation_metrics_svm = {
        'accuracy': svm_accuracy,
        'precision': svm_precision,
        'recall': svm_recall,
        'f1_score': svm_f1
    }
    append_evaluation_metrics_to_csv(model_name, evaluation_metrics_svm, dataset_name,best_params,fit_time)


if __name__ == "__main__":
    main()