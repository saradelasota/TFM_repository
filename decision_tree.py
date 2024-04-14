#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:51:45 2024

@author: saradelasota
"""


import pandas as pd
import time as t
import json

#TREE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, make_scorer

class DecisionTree:

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, criterion='entropy',rs=123):
        '''
        Initialize the decision tree
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required for a split
        :param random_state: random state
        :param criterion: measure function to use for best_partition, default to entropy
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.rs =rs
        self.tree = None

    def fit(self, X_train, y_train):
        '''
        Fit the decision tree to the training data.
        :param X: training features
        :param y: training labels
        '''
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state= self.rs
        )
        self.tree.fit(X_train, y_train)

    def predict(self, X_train):
        '''
        Make predictions using the trained decision tree.
        :param X: features of the data to be predicted
        :return: array of predicted labels
        '''
        if self.tree is None:
            raise ValueError("Decision tree has not been trained yet. Call fit() first.")

        predictions = self.tree.predict(X_train)
        return predictions

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
   
    # Define the parameters grid to search
    param_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'criterion':['gini','entropy']
    }
    
    # Define the scoring metric
    scorer = make_scorer(accuracy_score)
    
    # Instantiate the decision tree classifier
    decision_tree = DecisionTreeClassifier(random_state=rs)
    
    # Instantiate GridSearchCV
    grid_search = GridSearchCV(decision_tree, param_grid, scoring=scorer, cv=5)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("Best parameters found:", best_params)
    print("Best accuracy score:", best_score)
    
    # Instantiate the DecisionTree object with desired parameters
    decision_tree_model = DecisionTree(**best_params,rs=rs)
    
    start_time = t.time()

    # Fit the model to the training data
    decision_tree_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = decision_tree_model.predict(X_test)
    
    # Calculate time taken
    end_time = t.time()
    fit_time = end_time - start_time
    
    
    # Evaluate the model
    model_name = 'DecisionTree'
    dt_accuracy = accuracy_score(y_test, predictions)
    dt_precision = precision_score(y_test, predictions)
    dt_recall = recall_score(y_test, predictions)
    dt_f1 = f1_score(y_test, predictions)
    
    
    print("Test accuracy:", dt_accuracy)
    print("Test precision:", dt_precision)
    print("Test recall:", dt_recall)
    print("Test f1 score:", dt_f1)
    
    
    evaluation_metrics = {
    'accuracy': dt_accuracy,
    'precision': dt_precision,
    'recall': dt_recall,
    'f1_score': dt_f1
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
