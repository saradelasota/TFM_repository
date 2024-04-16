#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:06:27 2024

@author: saradelasota
"""

#IMPORTED libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    
    
    ##### COMPLETE UNSCALED DATASET ################
    df_complete_unscaled = pd.read_csv('dataset_TFM.csv')
    
    # Define a dictionary to map string values to binary values
    status_mapping = {'phishing': 0, 'legitimate': 1}

    df_complete_unscaled['status'] = df_complete_unscaled['status'].map(status_mapping)
    
    df_complete_unscaled.to_csv('datasets/df_complete_unscaled.csv')
    
    ##### LEXICAL UNSCALED DATSET #################
    #Drop columuns
    content_features = ['nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks','ratio_nullHyperlinks',
    'nb_extCSS','ratio_intRedirection','ratio_extRedirection','ratio_intErrors','ratio_extErrors',
    'login_form','external_favicon','links_in_tags','submit_email','ratio_intMedia','ratio_extMedia',
    'sfh','iframe','popup_window','safe_anchor','onmouseover','right_clic','empty_title',
    'domain_in_title','domain_with_copyright']

    external_features = ['whois_registered_domain','domain_registration_length','domain_age',
    'web_traffic','dns_record','google_index','page_rank']
        
    
    df_lexical_unscaled = df_complete_unscaled.drop(content_features + external_features, axis=1, inplace=False)
    
    df_lexical_unscaled.to_csv('datasets/df_lexical_unscaled.csv')


    ##### LEXICAL SCALED DATASET ################
    rs = 123
    
    # Split the data into training and testing sets
    X_train, X_test  = train_test_split(df_lexical_unscaled, test_size=0.2, random_state=rs)
       
    numerical_features = [
    'length_url', 'length_hostname', 'nb_dots', 'nb_hyphens', 'nb_at', 
    'nb_underscore', 'nb_percent', 'nb_slash', 'ratio_digits_url', 'nb_star','nb_colon',
    'ratio_digits_host', 'nb_subdomains', 'shortest_words_raw', 'length_words_raw',
    'shortest_word_host', 'shortest_word_path', 'longest_words_raw', 
    'longest_word_host', 'longest_word_path', 'avg_words_raw', 'char_repeat',
    'avg_word_host', 'avg_word_path','nb_qm', 'nb_and', 'nb_eq', 'nb_tilde', 'nb_comma', 
    'nb_semicolumn', 'nb_dollar', 'nb_www', 'nb_com', 
    'nb_dslash','nb_external_redirection','nb_redirection']
    
    # Scale the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled = scaler.transform(X_test[numerical_features])
    

    # Create DataFrames for scaled numerical features
    df_train_scaled_numerical = pd.DataFrame(X_train_scaled, columns=numerical_features)
    df_test_scaled_numerical = pd.DataFrame(X_test_scaled, columns=numerical_features)
    
    # Concatenate scaled numerical features with non-numerical features for both training and test sets
    df_train_scaled = pd.concat([X_train.drop(columns=numerical_features), df_train_scaled_numerical], axis=1)
    df_test_scaled = pd.concat([X_test.drop(columns=numerical_features), df_test_scaled_numerical], axis=1)
    
    # Concatenate training and test sets to create a unique dataset
    df_scaled = pd.concat([df_train_scaled, df_test_scaled], ignore_index=True)
    
    # Save the complete scaled dataset to a CSV file
    df_scaled.to_csv('datasets/df_lexical_scaled.csv', index=False)
    
    
    
    # Concatenate scaled numerical features with non-numerical features
    #df_scaled = pd.concat([df_lexical.drop(numerical_features, axis=1), df_scaled_numerical], axis=1)

    # Save the scaled dataset to a CSV file
    #df_scaled.to_csv('datasets/scaled_dataset.csv', index=False)
    

    
    ###### OTHER DATASETS ###########
    #FEATURE ENG
    df_selection1 = df_lexical_unscaled.drop(['nb_or'], axis=1)
    df_selection1.to_csv('datasets/df_selection1.csv', index=False)
    
    df_selection2 = df_lexical_unscaled.drop(['nb_or', 'nb_space', 'port', 'path_extension'], axis=1)
    df_selection2.to_csv('datasets/df_selection2.csv', index=False)

    
    

    
if __name__ == "__main__":
    main()












