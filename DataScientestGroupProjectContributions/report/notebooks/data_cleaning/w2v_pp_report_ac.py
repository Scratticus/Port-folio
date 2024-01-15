"""
DO NOT RUN THIS NOTEBOOK

This code is included to show hopw the full Word 2 vec csv was created, it requires a lot of time, CPU, disc space and RAM to complete.
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim

import re

def remove_text_between_tags(text):
    if isinstance(text, list):
        # If 'reviewText' is a list, apply the function to each element of the list
        return [re.sub(r'<.*?>.*?<.*?>', '', item) if isinstance(item, str) else item for item in text]
    elif isinstance(text, str):
        # If 'reviewText' is a string, apply the function directly
        return re.sub(r'<.*?>.*?<.*?>', '', text)
    else:
        return text
    
# -----------------------------------------
# Importing and sectioining Appliances json
# -----------------------------------------
    
full_df = pd.read_json("../../../data/processed/Appliances.json", lines = True)

# full_df.info()

full_df = full_df.dropna(subset=['reviewText'])

full_df.isna().sum()

print(full_df['overall'].value_counts())

# full_df.info()

full_df['overall'].unique()

full_df = full_df.sample(frac=1, random_state=69).reset_index()

# -----
# CSV 1
# -----

full_df_1 = full_df[:50000].reset_index()

print(full_df_1['overall'].value_counts())

features_1 = full_df_1['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_1 = full_df_1['overall']

features_1 = features_1.to_numpy().reshape(-1, 1)

# -----
# CSV 2
# -----

full_df_2 = full_df[50000:100000].reset_index()

print(full_df_2['overall'].value_counts())

features_2 = full_df_2['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_2 = full_df_2['overall']

features_2 = features_2.to_numpy().reshape(-1, 1)

# -----
# CSV 3
# -----

full_df_3 = full_df[100000:150000].reset_index()

print(full_df_3['overall'].value_counts())

features_3 = full_df_3['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_3 = full_df_3['overall']

features_3 = features_3.to_numpy().reshape(-1, 1)

# -----
# CSV 4
# -----

full_df_4 = full_df[150000:200000].reset_index()

print(full_df_4['overall'].value_counts())

features_4 = full_df_4['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_4 = full_df_4['overall']

features_4 = features_4.to_numpy().reshape(-1, 1)

# -----
# CSV 5
# -----

full_df_5 = full_df[200000:250000].reset_index()

print(full_df_5['overall'].value_counts())

features_5 = full_df_5['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_5 = full_df_5['overall']

features_5 = features_5.to_numpy().reshape(-1, 1)

# -----
# CSV 6
# -----

full_df_6 = full_df[250000:300000].reset_index()

print(full_df_6['overall'].value_counts())

features_6 = full_df_6['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_6 = full_df_6['overall']

features_6 = features_6.to_numpy().reshape(-1, 1)

# -----
# CSV 7
# -----

full_df_7 = full_df[300000:350000].reset_index()

print(full_df_7['overall'].value_counts())

features_7 = full_df_7['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_7 = full_df_7['overall']

features_7 = features_7.to_numpy().reshape(-1, 1)

# -----
# CSV 8
# -----

full_df_8 = full_df[350000:400000].reset_index()

print(full_df_8['overall'].value_counts())

features_8 = full_df_8['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_8 = full_df_8['overall']

features_8 = features_8.to_numpy().reshape(-1, 1)

# -----
# CSV 9
# -----

full_df_9 = full_df[400000:450000].reset_index()

print(full_df_9['overall'].value_counts())

features_9 = full_df_9['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_9 = full_df_9['overall']

features_9 = features_9.to_numpy().reshape(-1, 1)

# ------
# CSV 10
# ------

full_df_10 = full_df[450000:500000].reset_index()

print(full_df_10['overall'].value_counts())

features_10 = full_df_10['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_10 = full_df_10['overall']

features_10 = features_10.to_numpy().reshape(-1, 1)

# ------
# CSV 11
# ------

full_df_11 = full_df[500000:550000].reset_index()

print(full_df_11['overall'].value_counts())

features_11 = full_df_11['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_11 = full_df_11['overall']

features_11 = features_11.to_numpy().reshape(-1, 1)

# ------
# CSV 12
# ------

full_df_12 = full_df[550000:].reset_index()

print(full_df_12['overall'].value_counts())

features_12 = full_df_12['reviewText'].apply(lambda x: remove_text_between_tags(x))

target_12 = full_df_12['overall']

features_12 = features_12.to_numpy().reshape(-1, 1)

# ----------------------------
# feature/target List Creation
# ----------------------------

feature_list = [features_1, features_2, features_3, features_4, features_5, features_6, features_7, features_8, features_9, features_10, features_11, features_12]
target_list = [target_1, target_2, target_3, target_4, target_5, target_6, target_7, target_8, target_9, target_10, target_11, target_12]

# --------------------------------
# Creation of df portions as files
# --------------------------------

with tqdm(total=12) as pbar1:
    for i, feat_samp in enumerate(feature_list):
        
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

        review_list = []
        for review in feat_samp:
            if isinstance(review[0], str):
                tokens = tokenizer.tokenize(review[0].lower())
                review_string = ''
                for word in tokens:
                    review_string += word + ' '

                review_list.append(review_string) 

        review_series = pd.Series(review_list)

        vectorizer = CountVectorizer(stop_words='english')

        X = vectorizer.fit_transform(review_series)

        # Make a dataframe for machine learning
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        
        filename = f'../../../data/processed/vocab_df_part_{i}.csv'
        df.to_csv(filename, index=False)

        pbar1.update(1)

# ----------------------
# Creation of Vocab list
# ----------------------

filepaths = [
    '../../../data/processed/vocab_df_part_0.csv',
    '../../../data/processed/vocab_df_part_1.csv',
    '../../../data/processed/vocab_df_part_2.csv',
    '../../../data/processed/vocab_df_part_3.csv',
    '../../../data/processed/vocab_df_part_4.csv',
    '../../../data/processed/vocab_df_part_5.csv',
    '../../../data/processed/vocab_df_part_6.csv',
    '../../../data/processed/vocab_df_part_7.csv',
    '../../../data/processed/vocab_df_part_8.csv',
    '../../../data/processed/vocab_df_part_9.csv',
    '../../../data/processed/vocab_df_part_10.csv',
    '../../../data/processed/vocab_df_part_11.csv'
]

vocab = []

for filepath in filepaths:
    headers = pd.read_csv(filepath, nrows=0).columns.tolist()

    vocab.extend(headers)

vocab = list(set(vocab))

print(len(vocab))

# --------------------------------------
# Google Word 2 Vec model initialisation
# --------------------------------------

GOOGLEMODEL = gensim.models.KeyedVectors.load_word2vec_format('../../../data/processed/GoogleNews-vectors-negative300.bin', binary=True)

GOOGLEMODEL['div'].shape

# --------------------------------------------
# Countvector csv to Word 2 vec csv conversion
# --------------------------------------------

vocab_df = pd.DataFrame(columns=vocab)

with tqdm(total=len(filepaths)) as pbar:
    for i, filepath in enumerate(filepaths):
        temp_df = pd.read_csv(filepath)
        
        print(len(temp_df.columns), flush=True)

        common_columns = list(set(list(set(vocab_df.columns) & set(temp_df.columns))))

        temp_df = pd.merge(vocab_df, temp_df, on=common_columns, how='outer')

        temp_df = temp_df.fillna(0)

        print(len(temp_df.columns), flush=True)

        W2V_list = []

        for record in range(temp_df.shape[0]):
            
            sentence = np.zeros(300)

            indices = np.where(np.array(temp_df.iloc[record, :] >= 1))[0]

            for index in indices:
                word = vocab[index]
                if word in GOOGLEMODEL.key_to_index.keys():
                    sentence = sentence + GOOGLEMODEL[word]

            W2V_list.append(sentence)

        print('creating w2v_df', flush=True)

        W2V_df = pd.DataFrame(W2V_list)

        new_filepath = f'../../../data/processed/W2V_DF_{i}.csv'

        W2V_df.to_csv(new_filepath, index=False)

        pbar.update(1)

# ------------------------------------
# Concatentaion of features and target
# ------------------------------------

w2v_filepaths = [
    '../../../data/processed/W2V_DF_0.csv',
    '../../../data/processed/W2V_DF_1.csv',
    '../../../data/processed/W2V_DF_2.csv',
    '../../../data/processed/W2V_DF_3.csv',
    '../../../data/processed/W2V_DF_4.csv',
    '../../../data/processed/W2V_DF_5.csv',
    '../../../data/processed/W2V_DF_6.csv',
    '../../../data/processed/W2V_DF_7.csv',
    '../../../data/processed/W2V_DF_8.csv',
    '../../../data/processed/W2V_DF_9.csv',
    '../../../data/processed/W2V_DF_10.csv',
    '../../../data/processed/W2V_DF_11.csv'
]

w2v_df_dict = {}

for i, path in enumerate(w2v_filepaths):
    w2v_df_dict[i] = pd.read_csv(path)
    w2v_df_dict[i] = pd.concat([w2v_df_dict[i], target_list[i]], axis=1)

# ----------------------------------
# Crestion of full Word2vec Csv file
# ----------------------------------

w2v_df = pd.DataFrame()

for i, df in w2v_df_dict.items():
    w2v_df = pd.concat([w2v_df, df], axis = 0)

print(w2v_df.info())

w2v_df.to_csv('../../../data/processed/full_w2v_df')