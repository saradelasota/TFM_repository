#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:53:58 2024

@author: saradelasota
"""
import pandas as pd
import json


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
