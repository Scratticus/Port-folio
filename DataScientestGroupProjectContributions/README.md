Text Review Sentiment Analysis using Classification Modelling vs Regression Modelling 
==============================

Contributors
------------
Contributors:  
* Adam Cleaver BEng, 
* Ing. Lukas Topinka, M.Sc. 

DataScientest Supervisor: 
* Maëlys Bassilekin Blanc 

Project Description
------------
This Project takes review text from Amazon reviews and uses various machine learning models to explore how accurately the rating can be predicted by machine learning models. This serves as an introductory investigation into neuro linguistic processing (NLP) to create sentiment analyses. The project looks at the specific strengths of regression models and classification models against review rating accuracy. 

While the project ultimately did not generate a highly precise tool for predicting amazon review sentiments, a lot of learnings were taken from the various stages of preprocessing and machine learning models utilized. Enabling scrutinization of the classification and regression models and their benefits in potential use cases.  

The project builds a solid foundation for further investigation and suggests content to support a potential roadmap for further preprocessing techniques and more complex machine learning or deep learning models for future investigations. 

In this report several Classification and Regression Models are compared using data from amazon reviews as detailed in the Data Quality Report. The goal of the modelling is to accurately analyze the sentiment of the text review and predict the rating that the user submitted alongside their review. 

Regression Models being compared include: 
* Linear Regression
* Lasso
* Ridge
* ElasticNet
* Histogram Gradient Boosting Regressor 

Classification Models being compared include: 
* LogisticRegression
* Support Vector Machine Classification
* K Nearest Neighbors
* Decision Tree Classifier
* Random Forest Classifier
* Naive Bayes
* Histogram Gradient Boosting Classifier

There are no releases associated with this project, installing and running the various notebooks is recommended only for educational purposes, to test the findings established in the report PDFs.

Project Organization
------------
amazon_review

    ├── README.md               <- This file
    ├── requirements.txt        <- all source requirements for install
    ├── LICENSE                 <- GNU General Public License
    ├── .gitignore
    ├── data               
    │   ├── processed           <- Final vocabularies available in google drive. https://drive.google.com/drive/folders/1MMH-EXHOmrBhmsoCN14KgrqF9OsVWXmy?usp=sharing
    │   └── raw                 <- Raw data JSONs can be found here: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ 
    │
    ├── models                  <- Final models available in google drive. https://drive.google.com/drive/folders/1MMH-EXHOmrBhmsoCN14KgrqF9OsVWXmy?usp=sharing
    |   └── classification      <- Classification models location
    |   └── regression          <- Regression models location
    │
    ├── report          
    |   └── images                  <- location for all generated images
    |   └── notebooks
    |       └── classification      <- notebooks generating classification models
    |       └── data_cleaning       <- notebooks for cleaning datasets
    |           └── lem_stem_functions      <- Houses Text processes which can be called from any location in the project.
    |       └── deep_learning       <- WIP notebook generating deep learning models
    |       └── feature_extraction  <- WIP notebook generating feature extraction models and new vocab lists
    |       └── regression          <- notebooks generating regression models
    |   └── PDFs                <- location for PDF report versions   
    |   └── streamlit_app       <- streamlit_app location. streamlit app should be run from containing folder.

Data Source
-------------
Justifying recommendations using distantly-labeled reviews and fined-grained aspects 

Jianmo Ni, Jiacheng Li, Julian McAuley 

Empirical Methods in Natural Language Processing (EMNLP), 2019 

https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/ 


Using this Project
--------------
This Project is intended for educational use to the end goal of inspiring meaningful contributions to Neuro Linguistic Processing method technologies, as such, the most valuable information is summarized in the PDF files in the report directory, where options for future work, and hypotheses are detailed.

Users wishing to run the code themselves for creating vocab lists or prediction models can simply install from the requirements.txt and run the files of interest individually. TO use the models or vocabularies generated during the creation of the project users will need to download them from a separate google drive location (due to GitHub filesize limits.)