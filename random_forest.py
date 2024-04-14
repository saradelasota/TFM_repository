#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:17:03 2024

@author: saradelasota
"""

import pandas as pd
import time as t
import json

#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class RandomForest:
    
    def __init__(self, n_estimators=100, max_depth=None, criterion=None,max_features=None, rs=123):
        '''
        Initialize the Random Forest model
        :param n_estimators: The number of trees in the forest
        :param max_depth: The maximum depth of the tree
        :param random_state: Controls both the randomness of the bootstrapping of the samples used when building trees
                              (if `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node.
        :param criterion:
        :param max_features:
        '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.rs = rs
        self.rf = None
        self.feature_importance_scores = None


    def fit(self, X_train, y_train):
        '''
        Fit the Random Forest model to the training data
        :param X_train: The training features
        :param y_train: The training labels
        '''
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion= self.criterion,
                                             max_depth=self.max_depth, max_features=self.max_features,
                                             random_state= self.rs)
        self.rf.fit(X_train, y_train)
        self.feature_importance_scores = self.rf.feature_importances_
        
    def predict(self, X_test):
        '''
        Make predictions using the trained Random Forest model
        :param X_test: The test features
        :return: Predicted labels
        '''
        if self.rf is None:
            raise ValueError("Random Forest model has not been trained yet. Call fit() first.")
        return self.rf.predict(X_test)


def main():
    df = pd.read_csv('scaled_dataset.csv')
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
    param_grid_rf = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 5, 10],         # Maximum depth of the trees
        'criterion': ['gini', 'entropy'],  # Split criterion
        'max_features': ['auto', 'sqrt']   # Maximum number of features considered for splitting
    }
    
    
    #Initiate the RandomForestModel
    random_forest_model = RandomForestClassifier(random_state = rs)
    
    # Instantiate GridSearchCV
    grid_search_rf = GridSearchCV(random_forest_model, param_grid_rf, cv=5)
    
    # Fit the grid search to the data
    grid_search_rf.fit(X_train, y_train)
    
    # Get the best parameters and best score
    best_params = grid_search_rf.best_params_
    best_score = grid_search_rf.best_score_
    
    print("Best parameters found:", best_params)
    print("Best accuracy score:", best_score)

    # Instantiate the DecisionTree object with desired parameters
    #aqui poner #**best_params  
    rf_model = RandomForest(**best_params,rs=rs)
    
    start_time = t.time()

    # Fit the model to the training data
    rf_model.fit(X_train, y_train)
    
    
    # Make predictions on the test data
    predictions = rf_model.predict(X_test)
    
    # Calculate time taken
    end_time = t.time()
    
    
    # Evaluate the model
    model_name = "RandomForest"
    rf_accuracy = accuracy_score(y_test, predictions)
    rf_precision = precision_score(y_test, predictions)
    rf_recall = recall_score(y_test, predictions)
    rf_f1 = f1_score(y_test, predictions)
    
    
    fit_time = end_time - start_time
    
    
    print("Test accuracy:", rf_accuracy)
    print("Test precision:", rf_precision)
    print("Test recall:", rf_recall)
    print("Test f1 score:", rf_f1)
    
    
    evaluation_metrics = {
    'accuracy': rf_accuracy,
    'precision': rf_precision,
    'recall': rf_recall,
    'f1_score': rf_f1
    }
    
    # Call the function to append the new evaluation metrics to the existing CSV file
    append_evaluation_metrics_to_csv(model_name, evaluation_metrics,dataset_name,best_params, fit_time)
    
    
def append_evaluation_metrics_to_csv(model_name, evaluation_metrics,dataset_name,best_params,fit_time, filename='model_evaluation_metrics.csv'):
    try:
        # Load existing CSV file
        df_metrics = pd.read_csv(filename)
    except FileNotFoundError:
        # If the file doesn't exist yet, create an empty DataFrame
        df_metrics = pd.DataFrame(columns=['Model_Name','Dataset','Parameters','Accuracy', 'Precision', 'Recall', 'F1_Score', 'Fit_Time'])

    # Convert the best_params dictionary to a JSON string
    best_params_str = json.dumps(best_params)


    # Create a DataFrame with the new metrics
    new_row = pd.DataFrame([[model_name,dataset_name,best_params_str, evaluation_metrics['accuracy'], evaluation_metrics['precision'],
                             evaluation_metrics['recall'], evaluation_metrics['f1_score'], fit_time]],
                           columns=['Model_Name', 'Dataset','Parameters','Accuracy', 'Precision', 'Recall', 'F1_Score', 'Fit_Time'])

    # Append the new row to the existing DataFrame
    df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df_metrics.to_csv(filename, index=False)

if __name__ == "__main__":
    main()