#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:06:27 2024

@author: saradelasota
"""

#IMPORTED libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    
    full_df = pd.read_csv('dataset_TFM.csv')
    
    #Drop columuns
    
    content_features = ['nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks','ratio_nullHyperlinks',
    'nb_extCSS','ratio_intRedirection','ratio_extRedirection','ratio_intErrors','ratio_extErrors',
    'login_form','external_favicon','links_in_tags','submit_email','ratio_intMedia','ratio_extMedia',
    'sfh','iframe','popup_window','safe_anchor','onmouseover','right_clic','empty_title',
    'domain_in_title','domain_with_copyright']

    external_features = ['whois_registered_domain','domain_registration_length','domain_age',
    'web_traffic','dns_record','google_index','page_rank']
    
    full_df.head()
    print(type(full_df))
    df_lexical = full_df.drop(content_features + external_features, axis=1, inplace=False)

    #df_lexical = df_lexical.drop(external_features, axis=1, inplace=True)
    
    # Define a dictionary to map string values to binary values
    status_mapping = {'phishing': 0, 'legitimate': 1}

    # Map the 'status' variable using the dictionary
    df_lexical['status'] = df_lexical['status'].map(status_mapping)
    
    #df_lexical.to_csv('datasets/df_lexical.csv')

    #FEATURE ENG
    df_selected = df_lexical.drop(['nb_or', 'nb_space', 'port', 'path_extension'], axis=1)

    
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
    scaled_numerical = scaler.fit_transform(df_selected[numerical_features])
    df_scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical_features)

    # Concatenate scaled numerical features with non-numerical features
    df_scaled = pd.concat([df_lexical.drop(numerical_features, axis=1), df_scaled_numerical], axis=1)

    # Save the scaled dataset to a CSV file
    #df_scaled.to_csv('datasets/scaled_dataset.csv', index=False)
    
    full_df['status'] = full_df['status'].map(status_mapping)
    df_complete_unscaled = full_df
    df_complete_unscaled.to_csv('datasets/df_complete_unscaled.csv')
    
    
if __name__ == "__main__":
    main()












