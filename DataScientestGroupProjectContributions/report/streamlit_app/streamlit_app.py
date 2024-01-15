import sys
sys.path.append('../notebooks/data_cleaning/lem_stem_functions')

from text_functions_new_vocabs_ac import new_column_lemmatizer, new_column_stemmatizer, new_count_vectorize_data, new_tfidf_vectorize_data


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import pickle
import joblib
import os

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('../../data/processed/exampleDF.csv')
    # df = pd.read_csv('../../data/train.csv')
    df['reviewTime'] = pd.to_datetime(df['reviewTime'])
    dfCV = pd.read_csv('../../data/processed/ReducedExampleRatings.csv')
    return df, dfCV
    # return df
@st.cache_data
def load_best_models():
    with open('../../models/classification/RFBestModel.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('../../models/regression/HGBRBestModel.pkl', 'rb') as file:
        hgbr_model = pickle.load(file)
    with open('../../models/classification/LogisticBestModel.pkl', 'rb') as file:
        logistic_model = pickle.load(file)
    return rf_model, hgbr_model, logistic_model

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")

df, dfCV = load_data()
rf_model, hgbr_model, logistic_model = load_best_models()

# Set the title and sidebar
st.title("Comparison of Text Mined Customer Review Rating Prediction Models ")
st.sidebar.title("Table of Contents")
pages = ["Project Goals", "DataSet Quality", "Machine Learning Methodologies", "Application", "Conclusion", "Discussion and Next Steps"]
page = st.sidebar.radio("Go to", pages)

if page == pages[0]:
    st.title('Project Goals')
    st.markdown('## Abstract')
    st.markdown('This report compares methods of datamining the text from Amazon customer reviews against \
            the accuracy of machine learning model generated rating predictions from 1 to 5 stars. \
            This report satisfies the business need to identify the best model to accurately classify \
            customer reviews into a rating. The report aims to save future studies time in computation \
            comparisons by finding the best preprocessing methods and the best models to \
            classify reports by rating.  ')
    st.markdown('The findings of this report could be implemented in several use cases:  ')
    st.markdown("* Generate an automated rating system, which offers customers a pre-generated \
            star rating based on the content of their review.   ")
    st.markdown("* Identify Reviews which are incorrectly rated, to remove them from further \
            analyses or submit them to further analyses.   ")
    st.markdown("""* Classify reviews which are no longer associated with their original rating,
    or reviews which are not part of a rating system.  
    * Sort reviews for customer service customer response Management to organize reviews by priority.  
    * Classify reviews for automated CRM Tools enabling automated responses to reviews based on predicted rating.
    * Identify and handle reviews which are given in bad faith, by identifying text that often features in bad \
        faith reviews or that does not match typical review text associated with ratings. """)
    st.markdown('## Introduction')
    st.markdown("This Project takes review text from Amazon reviews and uses various machine learning models to explore how accurately\
                the rating can be predicted by machine learning models. This serves as an introductory\
                investigation into neuro linguistic processing (NLP) to create sentiment analyses. \
                The project looks at the specific strengths of regression models and classification models\
                against review rating accuracy.")
    st.markdown("While the project ultimately did not generate a highly precise tool for predicting amazon review sentiments, \
                a lot of learnings were taken from the various stages of preprocessing and machine learning models utilized. \
                Enabling scrutinization of the classification and regression models and their benefits in potential use cases.")
    st.markdown("Furthermore, the project builds a solid foundation for further investigation and suggests content to support a \
                potential roadmap for further preprocessing techniques and more complex machine learning or deep learning models \
                for future investigations.")
    

if page == pages[1] :
    @st.cache_data
    def load_synopsis():
        # column_dict = {
        # "Column Heading": ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
        #                    'reviewerName', 'reviewText', 'summary', 'unixreviewTime', 'image'],
        # "Brief": ['TARGET – star ratings valued 1-5 ', 'Number of upvotes granted to the review ',
        #           'Identifies verified buyers', 'Datetime review was left', 'Customer UID',
        #           'Item reference number', 'String holding various item specifics', 'Review info',
        #           'FEATURE - Text based review', 'Potential secondary Feature – review summary',
        #           'Datetime for review in unix time', 'Review images if included in review'],
        # "Number of Null Records": [0, 537515, 0, 0, 0, 0, 464804, 15, 324, 128, 0, 593519]
        # }
        column_dict = {
        "Column Heading": ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style',
                           'reviewerName', 'reviewText', 'summary', 'unixreviewTime', 'image'],
        "Brief": ['[TARGET] rating of the product', 'helpful votes of the review',
                  'verified buyers', 'time of the review (raw)', 'ID of the reviewer, e.g. A2SUAM1J3GNN3B',
                  'ID of the product, e.g. 0000013714', 'a dictionary of the product metadata, e.g., "Format" is "Hardcover"', 'name of the reviewer',
                  '[FEATURE] text of the review', 'summary of the review',
                  'time of the review (unix time)', 'images that users post after they have received the product'],
        "Number of Null Records": [0, 537515, 0, 0, 0, 0, 464804, 15, 324, 128, 0, 593519]
        }
        column_df = pd.DataFrame(column_dict)
        return column_df
    
    @st.cache_data
    def ratings():
        rating_dict = {
            "overall": [5, 4, 3, 2, 1],
            "Percentage of Reviews": ['69%', '13%', '5%', '3%', '10%']
        }
        rating_df = pd.DataFrame(rating_dict)
        return rating_df
    
    @st.cache_data
    def lem_stem():
        lem = pd.read_csv('../../data/processed/lem_example.csv')
        stem = pd.read_csv('../../data/processed/stem_example.csv')
        return lem, stem

    st.title("Dataset Quality")
    st.markdown('## Data Source')
    st.markdown("This report analyses Amazon review data from the Appliances Category, \
            the data was originally collected in 2014 and most recently updated in 2018. \
            Though the data has been parsed for NLP usage, extra Data Cleaning and preprocessing \
            is required to enable the variety of modelling techniques that will be tested in \
            this report.  ")
    st.markdown("The feature variables will be derived from the review text of each review and the \
            target variable will be the rating from 1-5 stars. The data quality is verifired by\
            checking for duplicates and NaN values, various tokenization and vectorization techniques\
            are implemented to enable the machine learning models to accurately parse the data.  ")
    st.markdown("Other columns can also be processed to enable further investigations and project \
             expansions. Though these extra deliverables will depend on favourable project scope and timeline.")
    st.markdown("#### Source Citation:")
    st.markdown('The original dataset analysed in this project is taken from the research paper')
    st.markdown("""> **Justifying recommendations using distantly-labeled reviews and fined-grained aspects**  
> Jianmo Ni, Jiacheng Li, Julian McAuley  
> _Empirical Methods in Natural Language Processing (EMNLP), 2019_""")
    st.markdown('For more details on the source of the data, the full research paper can be accessed by the following address: *https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/*')
    st.markdown('From the extensive dataset provided in the previously cited paper, only the category **Appliances** was selected for \
             further analysis. This was based on the size of the given subcategory, ranging from 500,000 to 1 million records, \
             as it provides a careful balance between data volume and computational efficiency. The aim was to ensure the availability \
             of a sufficiently robust dataset for effective training and validation of machine learning models, while avoiding the \
             potential for long running times in excess of 24 hours.')
    st.markdown('## Data Overview')
    st.markdown('This dataset contains a collection of products and associated ratings and additional information within \
             the selected **Appliances** category. This initial view of the provided **.json** dataset with the **.head()** function provides an \
             initial understanding of the structure and content of the dataset.')
    st.dataframe(df, hide_index=True)
    st.markdown('The information provided in each column of the dataset is explained in the following table, containing also number of Null Records.')
    st.dataframe(load_synopsis(), hide_index=True)
    st.markdown('### Missing values')
    st.image('../../report/images/data_quality_missing_values_viz_lt.png')

    st.markdown('### Target Data Balance')
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    with col1:
        st.dataframe(ratings(), hide_index=True)
    with col2:
        st.image('../../report/images/data_quality_overall_rating_distribution_lt.png')
    st.markdown('The target column ‘overall’ is imbalanced, as shown by the graph above. This indicates \
             that the data will need to be sampled to increase classification accuracy. 69% of the \
             reviews awarded a rating of 5, which indicates that a very basic model that simply calls \
             every review 5 stars will be 69% accurate against this data.')
    st.markdown(f"""> ##### The baseline Accuracy is 69%
> assuming all reviews are assigned rating 5.""")
    st.markdown('### Distribution of verified and unverified Reviews')
    st.image('../../report/images/data_quality_verification_distribution_lt.png')
    st.markdown('### Reviews Over the Years')
    st.image('../../report/images/data_quality_reviews_by_year_lt.png')
    st.markdown('### Dataset Preprocessing')
    st.markdown('For further preprocessing, only the selected **Target** and **Features** columns will be used.')
    st.markdown('###### Selected Target and Features')
    st.dataframe(df[['overall', 'reviewText']], hide_index=True)
    st.markdown("To achieve the best possible results from the dataset it is essential to reduce and format\
            the dataset into a data type and format that enables the models to generate the best possible\
            accuracies.  ")
    st.markdown("The reviews include html tags for videos and images if any were included in the review. \
            This text must be removed, as must any text including numbers, misspellings must be converted \
            to the correct spelling and tokenised. To achieve these results, Regex, pyspellchecker and the \
            RegExTokenizer were used to pre process the text in each review In addition to this the reviews\
            were compared to nltk's English stopwords and stopwords were removed.")
    st.markdown('Several further methods were implemented to ready the dataset for modelling. These included \
            WordNetLemmatizer and the EnglishStemmer from the nltk.stem library. These models reduce the words to \
            roots of the word in different formats. Following tables and Wordclouds represent a comparison between two \
            text-processing methods used.')
    lem, stem = lem_stem()
    # st.markdown('Following tables and Wordclouds represent a comparison between two text-processing methods used.')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##### Lemmatized text')
        st.dataframe(lem, hide_index=True)
        st.image('../../report/images/data_pp_lem_wordcloud_lt.png')
    with col2:
        st.markdown('##### Stemmatized text')
        st.dataframe(stem, hide_index=True)
        st.image('../../report/images/data_pp_stem_wordcloud_lt.png')
    st.markdown('To enable machine learning on these stemming methods, \
            the datasets need to be converted to number vectors. This was achieved by using a further two models in the CountVectorizer \
            and TFIDF Vectorizer from sci-kit Learns text library.')
    st.markdown('##### Vectorized data')
    st.markdown('An example of vectorized data using **CountVectorizer**(max_features=100).')
    st.dataframe(dfCV, hide_index=True)
    st.markdown("The data was also vectorized using Google's Word2Vec model, which did not use the stemmers to produce \
            vectors.")
    
if page == pages[2]:
    @st.cache_data
    def Regression_Bests():
        regression_dict = {
            "Model": ['HGBR', 
                      'Ridge',
                      'Linear Regression',
                      'Lasso:',
                      'ElasticNet:'
            ],
            "Token Method": [
                'lemmatized',
                'lemmatized',
                'English Stemmer',
                'lemmatized',
                'lemmatized',
            ],
            "Vector Method": [
                'TFIDF Vector',
                'TFIDF Vector',
                'TFIDF Vector',
                'Count Vector',
                'Count Vector'
            ],
            "Sampler": [
                'None',
                'Smote',
                'Smote',
                'None',
                'RandomOverSampler',
            ],
            "Mean Train Accuracy": [0.604, 0.523, 0.697, 0.294, 0.303],
            "Mean Test Accuracy": [0.593, 0.382, 0.380, 0.280, 0.269],
            "Mean Train Precision": [0.740, 0.750, 0.802, 0.659, 0.680],
            "Mean Test Precision": [0.728, 0.643, 0.512, 0.628, 0.615],
            "Mean Train Recall": [0.604, 0.523, 0.697, 0.294, 0.303],
            "Mean Test Recall": [0.593, 0.382, 0.380, 0.280, 0.269],
            "Mean Train F1 Score": [0.642,0.562, 0.724, 0.322, 0.331],
            "Mean Test F1 Score": [0.632, 0.435, 0.423, 0.303, 0.300],
            "Mean Train R Squared": [0.563, 0.892, 0.838, 0.488, 0.507],
            "Mean Test R Squared": [0.509, 0.510, 0.162, 0.444, 0.423],
            "Mean Train Mean Squared Error": [0.747, 0.700, 0.465, 1.470, 1.415],
            "Mean Test Mean Squared Error": [0.829, 1.393, 2.384, 1.580, 1.641],
        }
        regression_df = pd.DataFrame(regression_dict)
        return regression_df
    
    @st.cache_data
    def Classification_Bests():
        classification_dict = {
            "Model": ['SVM', 
                      'Logistic Regression',
                      'Random Forest',
                      'HGBC',
                      'Decision Tree',
                      'Naive bayes',
                      'K Nearest Neighbor',
            ],
            "Token Method": [
                'lemmatized',
                'lemmatized',
                'lemmatized',
                'English Stemmer',
                'lemmatized',
                'English Stemmer',
                'English Stemmer',
            ],
            "Vector Method": [
                'TFIDF Vector',
                'TFIDF Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
                'Count Vector',
            ],
            "Sampler": [
                'None',
                'None',
                'None',
                'None',
                'None',
                'RandomUnderSampler',
                'RandomUnderSampler'
            ],
            "Mean Train Accuracy": [0.874, 0.769, 0.971, 0.782, 0.971, 0.523, 0.556],
            "Mean Test Accuracy": [ 0.770, 0.762, 0.753, 0.742, 0.683, 0.560, 0.484],
            "Mean Train Precision": [0.882, 0.720, 0.971, 0.752, 0.971, 0.517, 0.556],
            "Mean Test Precision": [0.725, 0.703, 0.708, 0.682, 0.663, 0.711, 0.660],
            "Mean Train Recall": [0.814, 0.769, 0.971, 0.782, 0.971, 0.523, 0.556],
            "Mean Test Recall": [0.770, 0.763, 0.753, 0.742, 0.683, 0.624, 0.660],
            "Mean Train F1 Score": [0.856, 0.715, 0.970, 0.745, 0.970, 0.514, 0.552],
            "Mean Test F1 Score": [0.709, 0.707, 0.684, 0.698, 0.673, 0.659, 0.541],
            "Mean Train R Squared": [0.762, 0.408, 0.955, 0.407, 0.954, 0.311, 0.232],
            "Mean Test R Squared": [0.410, 0.376, 0.267, 0.299, 0.158, 0.111, -0.214],
            "Mean Train Mean Squared Error": [0.403, 1.004, 0.076, 1.00, 0.076, 1.378, 1.537],
            "Mean Test Mean Squared Error": [0.996, 1.053, 1.24, 1.18, 1.421, 1.501, 2.050],
        }
        classification_df = pd.DataFrame(classification_dict)
        return classification_df
    
    @st.cache_data
    def concat_two_dfs_vertical():
        df1 = Regression_Bests()
        df2 = Classification_Bests()
        concat_df = pd.concat([df1, df2])
        train_df = concat_df[[
            'Model',
            'Token Method', 
            'Vector Method', 
            'Sampler', 
            "Mean Train Accuracy",
            "Mean Train Precision",
            "Mean Train Recall",
            "Mean Train F1 Score",
            "Mean Train R Squared",
            "Mean Train Mean Squared Error"]].copy()
        train_df['Accuracy Type'] = 'Train'
        train_df = train_df.rename(columns={"Mean Train Accuracy": "Accuracy",
            "Mean Train Precision": "Precision",
            "Mean Train Recall": "Recall",
            "Mean Train F1 Score": "F1 Score",
            "Mean Train R Squared": "R Squared",
            "Mean Train Mean Squared Error": "Mean Squared Error"})
        test_df = concat_df[[
            'Model',
            'Token Method', 
            'Vector Method', 
            'Sampler', 
            "Mean Test Accuracy",
            "Mean Test Precision",
            "Mean Test Recall",
            "Mean Test F1 Score",
            "Mean Test R Squared",
            "Mean Test Mean Squared Error"]].copy()
        test_df['Accuracy Type'] = 'Test'
        test_df = test_df.rename(columns={"Mean Test Accuracy": "Accuracy",
            "Mean Test Precision": "Precision",
            "Mean Test Recall": "Recall",
            "Mean Test F1 Score": "F1 Score",
            "Mean Test R Squared": "R Squared",
            "Mean Test Mean Squared Error": "Mean Squared Error"})
        concat_with_acc_type = pd.merge(train_df, test_df, how='outer')
        grouped_concat = concat_with_acc_type.groupby(['Model', 'Token Method', 'Vector Method', 'Sampler', 'Accuracy Type']).max()
        grouped_concat = grouped_concat[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R Squared', 'Mean Squared Error']]
        grouped_concat = grouped_concat.sort_values(by="Accuracy", ascending=False).reset_index()
        return grouped_concat
    
    @st.cache_data
    def Word2Vec_df():
        dfw2v_dict = {
            "Model": ['Logistic Regression',
                      'Random Forest'
            ],
            "Vector Method": [
                'Google Word 2 Vector',
                'Google Word 2 Vector'
            ],
            "Sampler": [
                'RandomOverSampler',
                'RandomOverSampler'
            ],
            "Mean Test Accuracy": [0.526, 0.651],
            "Mean Test Precision": [0.60, 0.55],
            "Mean Test Recall": [0.53, 0.65],
            "Mean Test Specificity": [0.70, 0.36],
            "Mean Test F1 Score": [0.56, 0.57],
            "Mean Test Geometric Mean Error": [0.57, 0.26],
            "Mean Test Mean Squared Error": ['-', 2.31]
        }
        dfw2v = pd.DataFrame(dfw2v_dict)
        return dfw2v
    
    @st.cache_resource()
    def plot_and_show_accuracy_by_model():
        data = concat_two_dfs_vertical()
        x_values = range(-1, 13)
        threshold = [0.69] * len(x_values)
        fig = sns.catplot(x='Model', y='Accuracy', hue='Accuracy Type', data=data, kind='bar', height=8, aspect=0.7, legend=False)
        ax = fig.ax
        plt.plot(x_values, threshold, color="red", linestyle="--", label="Baseline Accuracy")
        plt.xticks(rotation=90)
        plt.yticks(list(np.arange(0, 1.1, 0.1)))
        plt.xlim(-0.5,11.5)
        plt.title('Accuracies by Model')
        plt.grid(axis="y")
        plt.tight_layout()
        plt.legend()
        plt.savefig('../images/overview_all_model_accuracies_ac.png')
        return fig
    
    st.title("Modelling")
    st.markdown('In this report several Classification and Regression Models are compared using data \
            from amazon reviews as detailed in the Data Quality Report. The models are detailed in the following sections. \
            the goal of the modelling is to accurately analyse the sentiment of the text review and predict the rating that the \
            user submitted alongside their review.')
    st.image('../../report/images/overview_schema_tree_lt.png')
    st.markdown('## Regression Models')
    st.markdown("""The Models being compared include:
* Linear Regression
* Lasso
* Ridge
* ElasticNet 
* Histogram Gradient Boosting Regressor""")
    st.markdown('## Classification Models')
    st.markdown("""The Models being compared include:
* LogisticRegression
* Support Vector Machine Classification
* K Nearest Neighbors
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes
* Histogram Gradient Boosting Classifier""")
    st.markdown('## Hypothesis')
    st.markdown('The ratings target variable consists of integers that follow a linear related scale. \
            Theoretically these reatings should be able to return a reliable score in regression models as well \
            as classification models. This project aims to compare the accuracy of this statement using a \
            comprehensive number of preprocessing and machine learning modelling techniques.')
    st.markdown("The project will compare the best accuracies of the machine learning models detailed above using multiple \
                preprocessing methods. The expected result is a similarity between the accuracy and error between the \
                classification models and the regression models.")
    st.markdown("""To measure this variance between models, the following scores will be taken from the model predictions:
* Accuracy: The number of correct predictions as a percentage.
* Precision: The number of correct positive predictions as a percentage.*
* Recall: The number of correct positives as a percentage with respect to false negative predictions.*
* F1 Score: A metric that combines Precision and Recall to provide a balanced measure*
* R Squared: The Good ness of fit indicator where values between 0.75 and 1 indicate a strong regression.**
* Mean Squared Error: Error indicator which penalises large errors.""")
    st.markdown("*Precision, Recall and F1 Score are all calculated using a weighted average, the support for each class impacts the score.")
    st.markdown("**When R Squared returns a negative value, this indicates an incredibly poor regression, this is a known bug of the scorer.")
    st.markdown("## Considerations")
    st.markdown('The classification models return exact categories 1-5, the regression models will only \
            return a continuous series of results. This has its benefits; the spread of the data can be \
            analyzed in greater detail than the classification reports. The visibility of the data can be used \
            to identify edge cases and see how noise in the categories behaves, which is not possible in the \
            classification reports.')
    st.markdown('The HistGradientBoostingRegressor (HGBR) and the HistGradientBoostingClassifier (HGBC) were \
            chosen over the GradientBoostingRegressor (GBR) and the GradientBoostingClassifier (GBC) to reduce \
            runtimes whilst operating on reasonably sized datasets.')
    st.markdown('## Model Comparisons')
    st.dataframe(concat_two_dfs_vertical(), hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_and_show_accuracy_by_model(), use_container_width=True)
    with col2:
        st.write("")
    
    st.markdown('## Results Analysis')
    st.markdown("The results show that Classification techniques are more suited to the text data, however \
            none of the results are strong enough to clearly define the models as better than a basic model. The \
            best model produces only 77% accurate results with a large mean squared error of 0.996 on the test data. \
            The average std deviation covers 1 class to either side of the correct class and the \
            confusion Matrices show that this accuracy largely relies on assigning a large percentage of the data to \
            class 5.")
    st.markdown("77% Accuracy is only 8% better than a basic model that returns class 5 for every prediction and so this model \
            can not be considered very strong.")
    col1, col2 = st.columns([3,1])
    with col1:
        st.image('../../report/images/classification_svm_best_conf_mtrx_ac.png', caption='Best Model Accuracy (SVM) Confusion Matrix', use_column_width=True)
    with col2:
        st.write("")
    
    st.markdown('### Google Word 2 Vector Analysis')
    st.markdown("The google Word 2 Vector Vectorizer is a text processor that assesses word relationships to support the \
            google search engine. Though the model is designed for search engine modelling rather than sentiment \
            analysis, the model was used to preprocess the review text to identify if the connections between words \
            in the model could provide a better model accuracy.")
    st.markdown("The word 2 vec preprocessor works on complete sentences rather than tokens like the previous preprocessing \
            techniques used. Each word in the sentence is converted into a vector of length 300. The vectors for each \
            word are added together to make one vector with 300 features representing the sentence.")
    st.markdown("The main advantage of this text process is that 300 features are processed much much faster than \
            a sparse matrix or dense matrix containing between 10,0000 and 70,000 features, depending on the contributing \
            processes and the number of records used to train the dataset.")
    st.markdown("Due to time constraints in the project, only two models with low processing times were modelled with the \
            word2vec text process, and no word stemming was used.")
    st.dataframe(Word2Vec_df(), hide_index=True)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.image('../../report/images/classification_logistic_r_w2v_conf_mtrx_ac.png', caption='Logistic Regression Word 2 Vector Confusion Matrix', use_column_width=True)
    with col2:
        st.image('../../report/images/classification_random_forest_w2v_conf_mtrx_ac.png', caption='Random Forest Word 2 Vector Confusion Matrix', use_column_width=True)
    with col3:
        st.write("")
    
    
    st.markdown('The score has worsened in comparison to the target baseline accuracy of 69%. \
            This is most likely due to the mismatch of the search engine preprocessing application and the sentiment analysis \
            application that is being performed. It is possible to customize the Word 2 Vec process for individual use cases. \
            This is a good starting point for further analysis.')
    st.markdown("## A Note on Sampling")
    st.markdown("It is obvious that the dataset is imbalanced, analysis was also performed on 4 samplers to see \
            which would be the most beneficial to the models, however the model quality was too poor for the \
            sampling results to be valid and so sampling was not well rated. Instead models are attuned on raw \
            unless an improvement is seen through smapling and oversampling techniques are largely dropped due \
            to data/working memory limitations.")
    st.markdown('Although the samplers proved to be unuseful, each one had a gridsearch performed to identify the maximum \
            potential sample with the given parameters.')
    
    options = {
        'RandomOverSampler, RandomUndersampler, no sampling',
        'Synthetic Minority Over Sampling Technique',
        'Cluster Centroids'
    }

    sampler_display = st.radio('The tested parameters included:', (
        'RandomOverSampler, RandomUndersampler, no sampling',
        'Synthetic Minority Over Sampling Technique',
        'Cluster Centroids'
    ))
    if sampler_display == 'RandomOverSampler, RandomUndersampler, no sampling':
        st.markdown('For the simpler models a simple best accuracy with the selected Sampler was taken. \
                For models with several arguments to be tested, these were tested simultaneously with \
                the sampler gridsearch to save computation in later more complex tests. These parameters \
                were graphed when available.')
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBR Regression'))
        if sample_display == 'Linear Regression':
            st.markdown("""The best results were returned by the English Stemmer TFIDF Vector text processes which were sampled by the RandomOverSampler.  
* Accuracy: 0.525
* Mean Squared Error: 2.020""")
        elif sample_display == 'Lasso Regression':
            st.markdown("""The best Lasso results were provided by the lemmatized Count Vector processes with no sampling.  
* Accuracy: 0.284
* Mean Squared Error: 0.271""")
            st.markdown('Lasso alpha values between 0.001 and 0.3 were tested simultaneously.')
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_lasso_sampling_alphas_ac.png', use_column_width=True)
            with col2:
                st.write("")

        elif sample_display == 'Ridge Regression':
            st.markdown("""The best Ridge results were provided by:  
* Lemmatizer Count Vector with RandomOverSampler sampling:  
    * Accuracy: 0.557  
    * Mean Squared Error: 2.915  
* Lemmatizer TFIDF Vector with RandomOverSampler sampling:
    * Accuracy: 0.478
    * Mean Squared Error: 1.11""")
            st.markdown('Ridge alpha values between 0.001 and 0.3 were tested simultaneously. Though \
                    0.001 has the best accuracy, 0.3 was take forward as the Mean Squared Error nearly tripled \
                    at lower alphas.')
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_ridge_sampling_alphas_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'ElasticNet Regression':
            st.markdown("""The best ElasticNet results were provided by:  
* Lemmatizer Count Vector with RandomOverSampler sampling:  
    * Accuracy: 0.318  
    * Mean Squared Error: 1.406  
* Lemmatizer Count Vector with no sampling:  
    * Accuracy: 0.296  
    * Mean Squared Error: 1.577""")
            st.markdown('ElasticNet alpha values between 0.001 and 0.3 and L1 ratio values between 0.3 and 0.7 were \
                    tested simultaneously.')
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_enet_sampling_alpha_ac.png', use_column_width=True)
            with col2:
                st.image('../../report/images/regression_enet_sampling_l1_ratio_ac.png', use_column_width=True)
            
        elif sample_display == 'HGBR Regression':
            st.markdown("""The best HGBC results were provided by:  
* Lemmatizer TFIDF Vector with RandomOverSampler sampling:  
    * Accuracy: 0.499  
    * Mean Squared Error: 1.351  
* Lemmatizer TFIDF Vector with no sampling:  
    * Accuracy: 0.410  
    * Mean Squared Error: 1.351""")
            st.markdown('HGBC learning rate values between 0.1 and 0.5 and max depths between 50 and 1000 were \
                    tested simultaneously.')
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_hgbr_no_sampling_learn_rate_ac.png', use_column_width=True)
            with col2:
                st.image('../../report/images/regression_hgbr_no_sampling_max_depth_ac.png', use_column_width=True)
            
            
    elif sampler_display == 'Synthetic Minority Over Sampling Technique':
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBR Regression'))
        if sample_display == 'Linear Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer TFIDF Vector - 500 k_neighbors  
    * Accuracy: 0.566  
    * Mean Squared Error: 2.96  
* Lemmatizer TFIDF Vector - 1000 k_neighbors  
    * Accuracy: 0.535  
    * Mean Squared Error: 1.98""")
            st.markdown('Though 500 k_neighbors has the best accuracy, 1000 k_neighbors were taken forward as the \
                     Mean Squared Error and R Squared values (R Squared not shown) performed marginally better.')
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_linear_r_smote_sampling_k_neighbors_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'Lasso Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* English Stemmer TFIDF Vector - 500 k_neighbors  
    * Accuracy: 0.224  
    * Mean Squared Error: 1.61  
* Lemmatizer Count Vector - 1000 k_neighbors  
    * Accuracy: 0.211  
    * Mean Squared Error: 1.54""")
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_lasso_smote_sampling_k_neighbors.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'Ridge Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer TFIDF Vector - 1000 k_neighbors  
    * Accuracy: 0.490  
    * Mean Squared Error: 1.081""")
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_ridge_smote_sampling_k_neighbors_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'ElasticNet Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer Count Vector - 1000 k_neighbors  
    * Accuracy: 0.237  
    * Mean Squared Error: 1.489""")
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_enet_smote_sampling_k_neighbors_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'HGBR Regression':
            st.markdown("""Smote K_Neighbours were tested between 5 and 1000:  
* Lemmatizer Count Vector - 50 k_neighbors  
    * Accuracy: 0.486  
    * Mean Squared Error: 1.188  
* Lemmatizer TFIDF Vector - 250 k_neighbors  
    * Accuracy: 0.443  
    * Mean Squared Error: 1.184  
* Lemmatizer TFIDF Vector - 100 k_neighbors  
    * Accuracy: 0.0.446  
    * Mean Squared Error: 1.176""")
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_hgbr_smote_sampling_k_neighbors.png', use_column_width=True)
            with col2:
                st.write("")
            
    elif sampler_display == 'Cluster Centroids':
        st.markdown('Although it was suspected that the Cluster Centroids sampler would perform well\n \
                if the dataset responds well to classification techniques, the Cluster Centroids method had no \
                effect on most of the models, only the HGBR model showed any change, and even that change was insignificant.')
        sample_display = st.radio('Which model do you want to view?', (
            'Linear Regression',
            'Lasso Regression',
            'Ridge Regression',
            'ElasticNet Regression',
            'HGBR Regression'))
        if sample_display == 'Linear Regression':
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_linear_r_cc_sampling_n_clusters_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'Lasso Regression':
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_lasso_cc_sampling_n_clusters_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'Ridge Regression':
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_ridge_cc_sampling_n_clusters_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'ElasticNet Regression':
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_enet_cc_sampling_n_clusters_ac.png', use_column_width=True)
            with col2:
                st.write("")
            
        elif sample_display == 'HGBR Regression':
            col1, col2 = st.columns(2)
            with col1:
                st.image('../../report/images/regression_hgbr_cc_sampling_n_clusters_ac.png', use_column_width=True)
            with col2:
                st.write("")
            


if page == pages[3]:
    rf_model, hgbr_model, logistic_model = load_best_models()
    st.title("Applications")
    st.markdown('Understanding the sentiment of customers is an incredibly important part of customer relation management. \
            Even a basic model could significantly help a relations manager to identify and prioritize which customers to \
            respond to. When combined with a tool that identifies key items related to overall marketing plans such a tool \
            could greatly affect the success of the marketing strategy.')
    st.markdown('Additional use cases could include a tool which predicts the rating of a review either for the customer or more \
            for a company who have unrated reviews, either through lack of rating system or through data quality issues.')
    st.markdown("In industries where review bombing is common, this tool could also be used as part of an anomolie detection. \
                By focussing on the precise reviews, the review bombs can largely be ignored or classified.")
    st.markdown('# Example Model')
    model_type_selection = st.selectbox("Choose a model type:", ["Classification", "Regression"])
    if model_type_selection == "Classification":
        model_selection = st.selectbox('Use the model below to compare modelling results:', ["Logistic Regression", "Random Forest"])
        if model_selection == "Logistic Regression":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_tfidf_vectorize_data(lem_review)
                [user_result] = logistic_model.predict(cv_review)
                [[class_1_confidence, class_2_confidence, class_3_confidence, class_4_confidence, class_5_confidence]] = logistic_model.predict_proba(cv_review)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {user_result}')
                if user_result == 1:
                    st.markdown(f'#### Confidence: {round(class_1_confidence * 100, 1)}%')
                if user_result == 2:
                    st.markdown(f'#### Confidence: {round(class_2_confidence * 100, 1)}%')
                if user_result == 3:
                    st.markdown(f'#### Confidence: {round(class_3_confidence * 100, 1)}%')
                if user_result == 4:
                    st.markdown(f'#### Confidence: {round(class_4_confidence * 100, 1)}%')
                if user_result == 5:
                    st.markdown(f'#### Confidence: {round(class_5_confidence * 100, 1)}%')
        elif model_selection == "Random Forest":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_count_vectorize_data(lem_review)
                [user_result] = rf_model.predict(cv_review)
                [[class_1_confidence, class_2_confidence, class_3_confidence, class_4_confidence, class_5_confidence]] = rf_model.predict_proba(cv_review)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {user_result}')
                if user_result == 1:
                    st.markdown(f'#### Confidence: {round(class_1_confidence * 100, 1)}%')
                if user_result == 2:
                    st.markdown(f'#### Confidence: {round(class_2_confidence * 100, 1)}%')
                if user_result == 3:
                    st.markdown(f'#### Confidence: {round(class_3_confidence * 100, 1)}%')
                if user_result == 4:
                    st.markdown(f'#### Confidence: {round(class_4_confidence * 100, 1)}%')
                if user_result == 5:
                    st.markdown(f'#### Confidence: {round(class_5_confidence * 100, 1)}%')
    elif model_type_selection == "Regression":
        model_selection = st.selectbox('Use the model below to compare modelling results:', ["Hist Gradient Boosting Regressor"])
        if model_selection == "Hist Gradient Boosting Regressor":
            user_review = st.text_area("Enter your review here:", "I think this product is great! Well built and sturdy!")
            if st.button("Generate Prediction"):
                lem_review = new_column_lemmatizer(user_review)
                cv_review = new_tfidf_vectorize_data(lem_review)
                cv_dense = cv_review.toarray()
                [user_result] = hgbr_model.predict(cv_dense)
                st.markdown('### Model Prediction')
                st.markdown(f'#### Class: {round(user_result, 2)}')
    st.markdown("The original examples are displayed below for use with the tool.")
    st.dataframe(df[['overall', 'reviewText']], hide_index=True)

if page == pages[4]: 
    st.title('Conclusion')
    st.markdown("The machine learning models generated in this report failed to generate an accuracy for the suggested business cases. \
                At best they match or marginally improve upon the baseline target accuracy of 69%, at worst the models dramatically \
                reduce the accuracy below that target value. ")
    st.markdown("""> ###### The best Model Accuracy on test data was 77%
> ###### only an 8% increase on the baseline Accuracy.""")
    st.markdown("These models cannot be used to predict review ratings with a high degree of confidence. Extreme values can be predicted \
                with better confidence than more nuanced middle values. The nuance is lost in the modelling. Further improvements are \
                required to create a suitable model that provides confidence in the results provided. ")
    st.image('../../report/images/overview_schema_w_acc_lt.png', width=1000)
    st.markdown("The classification methods produced better scores on Test data for all measurements, disproving the hypothesis.")
    st.markdown("The best use case of this report is a supporting document for further progress on building a conclusive text sentiment analysis.")
    
if page == pages[5]: 
    st.title('Discussion and Next Steps')
    st.markdown("## Further Investigation")
    st.markdown("Combining preprocessing techniques and modelling techniques into a Deep Learning model is a sensible next step \
            to pursue. Additionally, since the input data is heavily imbalanced and the greatest prediction \
            improvements were seen by increasing the size of the dataset, it could be wise to expand and diversify the \
            dataset by sampling from other amazon categories to generate more distinction, or by using a big data\
            methodology to enable the models to be trained on datasets with millions of records.")
    st.markdown("""In the machine learning methodologies section two areas for improvement were identified:
* Improving Preprocessing, by implementing feature reduction
* Improving Modelling, by using more advanced modelling Techniques.""")
    st.markdown("## Continuing the Investigation")
    st.markdown("In light of the poor results, some additional work was undertaken as a contribution to the next steps of the project. \
                These investigations were heavily time constricted, so only the 'low hanging fruit' options were investigated.")
    st.markdown("### Improving Preprocessing")
    st.markdown("The greatest potential improvement in preprocessing lies in the perceived removal of diffusion. Sampling worsened the\
                scores across the board, and so it can be assumed that either the data contains a lot of unimportant features which are being falsely replicated, few important features\
                which are not being fairly replicated due to the accompanying noise, or a combination of the two. As such a feature extraction process should help the models\
                to extract the relationships between the most important features.")
    st.markdown("Ideally, a Recursive Feature Elimnation (RFE) with Cross Validation (RFECV) could be run on the vocabulary to identified the words that have the greatest validated\
                on the results. However, the RFECV takes a long time to run, each step trains the chosen estimator against the split dataset, tests the\
                results against the remainder of the dataset and all steps are repeated by the number of cross validations. On top of this there \
                is some computational conglomeration time to consider. For a detailed study this will take an extremely long time, more time than is realistically available in the tenure of this project.")
    st.markdown("Instead a simple RFE was run on the entire dataset by quarters. The chosen estimator was the Random Forest Estimator which had the best test accuracy scores.\
                A step of 1000 was implemented with a desired number of features set to 3000.")
    st.markdown("The 4 quarters produced a list of most important words which were merged to create a list of 3922 unique features. A reduction from 28,822 features to 3,922.")
    st.markdown("The extracted features were parsed through the custom lemmatizer function, instead of using English stop words as a no-go gauge, the function was inverted and the \
                important feature list was implemented as a go through gauge. In this way the model only considered words that were deemed important by the RFE process.")
    st.markdown("The reduced feature Random Forest accuracy and mean squared error scores did not improve. The accuracy score reduced to 67.8% and the mean squared error score increased to 2.30. \
                This indicates that this feature selection has hurt the modelling process more than it has improved it as the accuracy has now fallen below the baseline accuracy.")
    st.markdown("It is plausible that the RFE was applied overzealously, possibly reducing the feature variables too harshly, so that many records had little to no data\
                to build a prediction from.")
    st.markdown("Another potential avenue is the revisiting and customizing of the google word2vec vectorizer. This vectorizer measures whole sentences\
                and provides a relational summary of the sentence. Customizing the vectorizer to this sentiment analysis use case\
                could yield a substantial score improvement. Theoretically the model should be able to detect a lot of connections between the \
                words featured in the review that other modelling techniques may miss. For example 'not bad' is a positive or at least average \
                but the individual words could both potentially score a review negatively. The word 2 vec vectorizer should see past the unique words and\
                consider the relationship between them. This may also alleviate the lack of nuance in the models.")
    st.markdown("##### Alternate Preprocessing steps")
    st.markdown("During the investigation, the quality of the dataset was brought into question. Another avenue of exploration would be to expand the dataset fed to the models.\
                The ideal Dataset would consist of an amalgamation of categories from the amazon directory, using a sampling method which harvests a good distribution of target variables.\
                Ideally the data would be amalgamated into one dataset which is reasonably well balanced and of considereable size. Two million records would be a conservative estimate.\
                With this many records and more balance than present in the current dataset all function results should be improved, from preprocessing to modelling, including the \
                feature extraction attempted in this further work. Working with this much data will require mass data handling and a pySpark or similar solution would need to be implemented \
                to make analyses possible.")
    st.markdown('### Improving Modelling')
    st.markdown('Another analysis tool that has not yet been applied to the dataset is the **SHAP** (SHapley Additive exPlanations). \
                SHAP analysis is a powerful machine learning technique that explains the influence of individual features on the predictions \
                of a model. It can provide deeper insights into the relationships between words and their impact on the results. The following \
                figures show examples of SHAP analysis based on a part of the dataset, which is only 5%, using the CountVectoriser with a \
                maximum of 100 features. The calculation of this limited part took about 1832 minutes.')
    st.markdown('More complex models such as Deep Learning models could be implemented for such a purpose.')
    st.markdown('##### Overall SHAP analysis for all 5 classes')
    st.image('../../report/images/classification_shap_100_words_class_overview_lt.png')
    st.markdown('##### Separate SHAP analysis for each class')
    # Create columns for layout
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image('../../report/images/classification_shap_100_words_class_1_lt.png')
    with col2:
        st.image('../../report/images/classification_shap_100_words_class_2_lt.png')
    with col3:
        st.image('../../report/images/classification_shap_100_words_class_3_lt.png')
    with col4:
        st.image('../../report/images/classification_shap_100_words_class_4_lt.png')
    with col5:
        st.image('../../report/images/classification_shap_100_words_class_5_lt.png')
    st.markdown("### A Combined Approach - Deep Learning Hypothesis")
    st.markdown("A key advantage of the regression modelling results is the analytical potential of the uncut results. These results allow observations to be drawn from the distribution of \
                each class. In comparison, a key advantage of the classification models is the categorical output with accompanying model confidence. See the boxplot chart below for \
                a graphical representation of these advantages.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Regression test spread analysis.")
        st.image('../../report/images/regression_boxplot_pre_cut_test_data_ac.png', use_column_width=True)
    with col2:
        st.write("##### Classification test spread analysis.")
        st.image('../../report/images/classification_boxplot_test_data_ac.png', use_column_width=True)
    st.markdown("A potential ideal result could be provided by amalgamating a deep learning model which takes advantage of these benefits. \
                In such a model, several stages of processing would be performed on the text data before it is classified. \
                This would allow a combination of methods to be applied to the dataset which makes the most of the strengths of each stage.")
    st.markdown("""Example Deep Learning model: (This model does not consider a specific neural network framework, but instead an amalgamation of seen processes.)
* Initial Parse: Custom Word2Vec model investigating word relations.
    * Input: Text data as series.
    * Output: vector matrix with stems as headers.
* Initial reduction: Recursive feature extraction using the strongest regression model.
    * estimator: HistogramGradientBoosting Regressor.
    * scoring: neg_mean_squared_error
    * target features: 0.95
    * reduction targets the vocabulary rather than the dataset.
    * Output: masked vector matrix from strongest 95% vocabulary items.
* Secondary parsing: text data parsed using the previous stem and vector techniques.
    * output: merged matrix on headers grouped by stemmed headers
* Secondary Reduction: Same as initial reduction.
* Final Classification: Utilizing the strongest classifier.""")
    st.markdown("The Word2vec pass reduction pair can be looped together. The Secondary reduction can also be performed\
                multiple times until a satisfactory number of features is achieved. The key advantage of using a regression model\
                to reduce the important features is the precise mean squared error calculation. Ending the process with a classification\
                model still provides the user with a clear category and confidence interval of the prediction.")