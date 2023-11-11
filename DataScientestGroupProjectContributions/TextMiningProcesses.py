# Function to perform lemmatization and remove stopwords
def column_lemmatizer(text_series):
    """
    This function preprocesses a pandas Series of sentences, typically taken from a dataframe column and prepares them for classification/regression modelling
    by tokenizing, removing stop words, and lemmatizing the series.
    
    Args:
    text_series (pd.Series): Input pandas Series containing sentences.
    
    Returns:
    pd.Series: Processed Series containing lemmatized words.

    Example:
    df['to_be_lemmed'] = pd.Series({0: 'I like this', 1: 'good times'})
    df['lems'] = column_stemmatizer(df['to_be_lemmed'])

    returns:
    df['lems'] = pd.Series({0: ['like'], 1: ['good', 'time']})
    """
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize.regexp import RegexpTokenizer

    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')
        

    # Initialize the lemmatizer and stopwords set
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'[\w]+')

    lemmed_cells = []

    for i in text_series:
        
        #tokenize series to lower case
        tokens = tokenizer.tokenize(i.lower())

        # remove stop words
        stopped = []
        for token in tokens:
            if token not in stop_words:
                stopped.append(token)

        lemmed = ''
        for word in stopped:
            lemmed_word = lemmatizer.lemmatize(word)
            lemmed += lemmed_word + ' '

        lemmed.strip()

        lemmed_cells.append(lemmed)

    lemmed_series = pd.Series(lemmed_cells)

    return lemmed_series

def column_stemmatizer(text_series):
    """
    This function preprocesses a pandas Series of sentences, typically taken from a dataframe column and prepares them for classification/regression modelling
    by tokenizing, removing stop words, and stemmatizing the series.
    
    Args:
    text_series (pd.Series): Input pandas Series containing sentences.
    
    Returns:
    pd.Series: Processed Series containing stemmed words.

    Example:
    df['to_be_stemmed'] = pd.Series({0: 'I like this', 1: 'good times'})
    df['stems'] = column_stemmatizer(df['to_be_stemmed'])

    returns:
    df['stems'] = pd.Series({0: ['like'], 1: ['good', 'time']})
    """
    import pandas as pd
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.snowball import EnglishStemmer
    from nltk.tokenize.regexp import RegexpTokenizer

    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')

    # Initialize stemmer, stopwords and regex tokenizer
    stemmer = EnglishStemmer()
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'[\w]+')

    # prepare list to hold returned data
    stemmed_cells = []

    # Loop through columns:
    for i in text_series:

        # Tokenize series and set lower case
        tokens = tokenizer.tokenize(i.lower())

        # remove stopwords
        stopped = []
        for token in tokens:
            if token not in stop_words:
                stopped.append(token)
        
        stemmed = ''
        for word in stopped:
            stemmed_word = stemmer.stem(word)
            stemmed += stemmed_word + ' '
        
        stemmed.strip()

        stemmed_cells.append(stemmed)

    stemmed_series = pd.Series(stemmed_cells)

    return stemmed_series

def count_vectorize_data(X_train_processed, X_test_processed=None, max_features=None):
    """
    This function uses CountVectorizer to vectorize the train and test text datasets ready for sentiment analysis.
    
    The function takes preprocessed text data (X_train_processed and X_test_processed) as inputs,
    inputs are vectorized with the CountVectorizer function from scikit-learn to convert text into 
    numerical feature vectors, and returns the vectorized train and test data.
    
    Parameters:
    X_train_processed (pd.Series, column format, lemmatized/stemmatized): Preprocessed textual data for training.
    X_test_processed (pd.Series, column format, lemmatized/stemmatized): Preprocessed textual data for testing.
    
    Returns:
    train_X (scipy.sparse matrix): Vectorized training data.
    test_X (scipy.sparse matrix): Vectorized testing data.

    The sparse matrix type is handled by most classification/regression models without using the deprecated dense array types.

    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(max_features=max_features)

    if X_test_processed is not None and X_test_processed.any():
        train_X = vectorizer.fit_transform(X_train_processed)
        test_X = vectorizer.transform(X_test_processed)
        return train_X, test_X
    else: 
        train_X = vectorizer.fit_transform(X_train_processed)
        return train_X, None

    
def tfidf_vectorize_data(X_train_processed, X_test_processed=None, max_features=None):
    """
    Perform TF-IDF processing on the train and test text datasets ready for sentiment analysis.

    The function takes preprocessed text data (X_train_processed and X_test_processed) as inputs,
    inputs are vectorized with the TFID function from scikit-learn to convert text into 
    numerical feature vectors, and returns the vectorized train and test data.
    
    Parameters:
    X_train_processed (pd.Series, column format, lemmatized/stemmatized): Preprocessed textual data for training.
    X_test_processed (pd.Series, column format, lemmatized/stemmatized): Preprocessed textual data for testing.
    
    Returns:
    train_X (scipy.sparse matrix): Vectorized training data.
    test_X (scipy.sparse matrix): Vectorized testing data.

    The sparse matrix type is handled by most classification/regression models without using the deprecated dense array types.
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit and transform the specified text data
    if X_test_processed is not None and X_test_processed.any():
        train_X = tfidf_vectorizer.fit_transform(X_train_processed)
        test_X = tfidf_vectorizer.transform(X_test_processed)
        return train_X, test_X
    else: 
        train_X = tfidf_vectorizer.fit_transform(X_train_processed)
        return train_X, None
    
# # Test Elements

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# X = pd.Series({
#     0: 'good, I like it', 
#     1: 'happy best product',
#     2: 'good happy product',
#     3: 'bad stuff broken',
#     4: 'evil bad things',
#     5: 'good product'
# })

# lem_X = column_lemmatizer(X)

# print(lem_X)

# TFIDF_X = tfidf_vectorize_data(lem_X)

# print(TFIDF_X[0].shape)

# lr = LinearRegression()

# y = pd.Series({
#     0: 1, 
#     1: 1,
#     2: 1,
#     3: 0,
#     4: 0,
#     5: 0
# })

# lr.fit(TFIDF_X[0], y)

# print(lr.score(TFIDF_X[0], y))

