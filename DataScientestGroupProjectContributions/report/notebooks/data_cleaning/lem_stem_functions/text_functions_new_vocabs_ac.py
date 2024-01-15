def text_stemmer(tokens, stemmer, go_gauge):
    from nltk.stem import WordNetLemmatizer
    from nltk.stem.snowball import EnglishStemmer
    if stemmer == 'lemmatizer':
        lemmatizer = WordNetLemmatizer()
        stem_string = ''
        for token in tokens:
            if token:
                stemmed_word = lemmatizer.lemmatize(token)
                if stemmed_word in go_gauge:
                    stem_string += stemmed_word + ' '    
        stem_string = stem_string.strip()
        return stem_string
    elif stemmer == 'stemmatizer':
        stemmat = EnglishStemmer()
        stem_string = ''
        for token in tokens:
            if token:
                stemmed_word = stemmat.stem(token)
                if stemmed_word in go_gauge:
                    stem_string += stemmed_word + ' '    
        stem_string = stem_string.strip()
        return stem_string
    else:
        raise ValueError('Unsupported text stemmer passed')
        
def preprocess_items(text, tokenizer, spell_checker):
    import re
    import pandas as pd
    import nltk
    from nltk.tokenize.regexp import RegexpTokenizer

    if isinstance(text, str):
        item_no_tags = re.sub(r'<.*?>.*?<.*?>', '', text)
        item_tokens = tokenizer.tokenize(item_no_tags.lower())
        token_list = []
        for token in item_tokens:
            corrected_token = spell_checker.correction(token.strip())
            token_list.append(corrected_token)
        return token_list
    
def process_item(text, tokenizer, spell_checker, stemmer, go_gauge):
    tokens = preprocess_items(text=text, tokenizer=tokenizer, spell_checker=spell_checker)
    stemmed_string = text_stemmer(tokens=tokens, stemmer=stemmer, go_gauge=go_gauge)
    if stemmed_string is None:
        stemmed_string = ''
    return stemmed_string

def process_item_wrapper(args):
    item = args[0]
    tokenizer = args[1]
    spell_checker = args[2]
    stemmer = args[3]
    go_gauge = args[4]
    return process_item(text=item, tokenizer=tokenizer, spell_checker=spell_checker, stemmer=stemmer, go_gauge=go_gauge)
            
def new_column_lemmatizer(text_series):
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
    from nltk.tokenize.regexp import RegexpTokenizer
    from spellchecker import SpellChecker
    from tqdm import tqdm
    import csv

    from spellchecker import SpellChecker
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    spell_checker = SpellChecker()
    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')
  
    # Initialize the lemmatizer and stopwords set
    stop_words = set(stopwords.words('english'))
    try:
        with open('./data/processed/full_vocab_list.csv', 'r') as file:
            csv_list = file.read().strip()
            go_gauge = csv_list.split(',')
    except FileNotFoundError:
        try:
            with open('../data/processed/full_vocab_list.csv', 'r') as file:
                csv_list = file.read().strip()
                go_gauge = csv_list.split(',')
        except FileNotFoundError:
            try:
                with open('../../data/processed/full_vocab_list.csv', 'r') as file:
                    csv_list = file.read().strip()
                    go_gauge = csv_list.split(',')
            except FileNotFoundError:
                try:
                    with open('../../../data/processed/full_vocab_list.csv', 'r') as file:
                        csv_list = file.read().strip()
                        go_gauge = csv_list.split(',')
                except FileNotFoundError:
                    try:
                        with open('../../../../data/processed/full_vocab_list.csv', 'r') as file:
                            csv_list = file.read().strip()
                            go_gauge = csv_list.split(',')
                    finally:
                        raise FileNotFoundError('cannot find full_vocab_list.csv in the project tree.')


    if isinstance(text_series, str):
        if text_series.strip() != '':
            text_series = pd.Series([text_series])

    if isinstance(text_series, pd.Series):
        args_list = [
            (item, tokenizer, spell_checker, 'lemmatizer', go_gauge)
            for item in text_series
        ]

        # If 'reviewText' is a list, apply the function to each element of the list
        with ProcessPoolExecutor() as executor:
            lemmed_cells = list(executor.map(process_item_wrapper, args_list)
            )

            lemmed_series = pd.Series(lemmed_cells)

            return lemmed_series
    else:
        raise TypeError('function must take a pd.Series or a string as argument')

def new_column_stemmatizer(text_series):
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
    from nltk.tokenize.regexp import RegexpTokenizer
    from spellchecker import SpellChecker
    from tqdm import tqdm
    import csv

    from spellchecker import SpellChecker
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    spell_checker = SpellChecker()
    # Download NLTK resources
    if not nltk.corpus.stopwords.fileids():
        nltk.download('punkt')
        nltk.download('stopwords')
  
    # Initialize the stemmatizer and stopwords set
    stop_words = set(stopwords.words('english'))
    try:
        with open('./data/processed/full_vocab_list_stemmed.csv', 'r') as file:
            csv_list = file.read().strip()
            go_gauge = csv_list.split(',')
    except FileNotFoundError:
        try:
            with open('../data/processed/full_vocab_list_stemmed.csv', 'r') as file:
                csv_list = file.read().strip()
                go_gauge = csv_list.split(',')
        except FileNotFoundError:
            try:
                with open('../../data/processed/full_vocab_list_stemmed.csv', 'r') as file:
                    csv_list = file.read().strip()
                    go_gauge = csv_list.split(',')
            except FileNotFoundError:
                try:
                    with open('../../../data/processed/full_vocab_list_stemmed.csv', 'r') as file:
                        csv_list = file.read().strip()
                        go_gauge = csv_list.split(',')
                except FileNotFoundError:
                    try:
                        with open('../../../../data/processed/full_vocab_list_stemmed.csv', 'r') as file:
                            csv_list = file.read().strip()
                            go_gauge = csv_list.split(',')
                    finally:
                        raise FileNotFoundError('cannot find full_vocab_list_stemmed.csv in the project tree.')

    if isinstance(text_series, str):
        if text_series.strip() != '':
            text_series = pd.Series([text_series])

    if isinstance(text_series, pd.Series):
        args_list = [
            (item, tokenizer, spell_checker, 'stemmatizer', go_gauge)
            for item in text_series
        ]

        # If 'reviewText' is a list, apply the function to each element of the list
        with ProcessPoolExecutor() as executor:
            lemmed_cells = list(executor.map(
                process_item_wrapper, args_list
                )
            )

            lemmed_series = pd.Series(lemmed_cells)

            return lemmed_series
    else:
        raise TypeError('function must take a pd.Series as argument')
    
def new_count_vectorize_data(X_train_processed, X_test_processed=None, max_features=None):
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

    try:
        with open('./data/processed/full_vocab_list.csv', 'r') as file:
            csv_list = file.read().strip()
            full_vocab = csv_list.split(',')
    except FileNotFoundError:
        try:
            with open('../data/processed/full_vocab_list.csv', 'r') as file:
                csv_list = file.read().strip()
                full_vocab = csv_list.split(',')
        except FileNotFoundError:
            try:
                with open('../../data/processed/full_vocab_list.csv', 'r') as file:
                    csv_list = file.read().strip()
                    full_vocab = csv_list.split(',')
            except FileNotFoundError:
                try:
                    with open('../../../data/processed/full_vocab_list.csv', 'r') as file:
                        csv_list = file.read().strip()
                        full_vocab = csv_list.split(',')
                except FileNotFoundError:
                    try:
                        with open('../../../../data/processed/full_vocab_list.csv', 'r') as file:
                            csv_list = file.read().strip()
                            full_vocab = csv_list.split(',')
                    finally:
                        raise FileNotFoundError('cannot find full_vocab_list.csv in the project tree.')

    vectorizer = CountVectorizer(max_features=max_features, vocabulary=full_vocab)

    if X_test_processed is not None and X_test_processed.any():
        train_X = vectorizer.transform(X_train_processed)
        test_X = vectorizer.transform(X_test_processed)
        return train_X, test_X
    else: 
        train_X = vectorizer.transform(X_train_processed)
        return train_X

    
def new_tfidf_vectorize_data(X_train_processed, X_test_processed=None, max_features=None):
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

    try:
        with open('./data/processed/full_vocab_list.csv', 'r') as file:
            csv_list = file.read().strip()
            full_vocab = csv_list.split(',')
    except FileNotFoundError:
        try:
            with open('../data/processed/full_vocab_list.csv', 'r') as file:
                csv_list = file.read().strip()
                full_vocab = csv_list.split(',')
        except FileNotFoundError:
            try:
                with open('../../data/processed/full_vocab_list.csv', 'r') as file:
                    csv_list = file.read().strip()
                    full_vocab = csv_list.split(',')
            except FileNotFoundError:
                try:
                    with open('../../../data/processed/full_vocab_list.csv', 'r') as file:
                        csv_list = file.read().strip()
                        full_vocab = csv_list.split(',')
                except FileNotFoundError:
                    try:
                        with open('../../../../data/processed/full_vocab_list.csv', 'r') as file:
                            csv_list = file.read().strip()
                            full_vocab = csv_list.split(',')
                    finally:
                        raise FileNotFoundError('cannot find full_vocab_list.csv in the project tree.')

    # Create a TfidfVectorizer instance
    tfidf_vectorizer = TfidfVectorizer(vocabulary=full_vocab, max_features=max_features)

    # Fit and transform the specified text data
    if X_test_processed is not None and X_test_processed.any():
        train_X = tfidf_vectorizer.fit_transform(X_train_processed)
        test_X = tfidf_vectorizer.transform(X_test_processed)
        return train_X, test_X
    else: 
        train_X = tfidf_vectorizer.fit_transform(X_train_processed)
        return train_X

# # Test Elements



# Fit and transform the specified text data

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_extraction.text import CountVectorizer
    import csv

    with open('full_vocab_list.csv', 'r') as file:
        csv_list = file.read().strip()
        header_list = csv_list.split(',')

    # Create a TfidfVectorizer instance
    # vectorizer = CountVectorizer(vocabulary=header_list)

    X = pd.Series({
        0: 'good, I like it', 
        1: '<http. vanilla> happy best*&^#*(&t',
        2: 'ceramic certain juice rtequila',
        3: '',
        4: '9999',
        5: 'good product'
    })

    lem_X = new_column_lemmatizer(X)

    print(lem_X)

    vectors = new_tfidf_vectorize_data(lem_X)

    vector_df = pd.DataFrame(vectors.todense(), columns=header_list)

    print(vector_df.shape)

    lr = LinearRegression()

    y = pd.Series({
        0: 1, 
        1: 1,
        2: 1,
        3: 0,
        4: 0,
        5: 0
    })

    lr.fit(vectors, y)

    print(lr.score(vectors, y))

