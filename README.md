# P5-Intro-to-Machine-Learning
Udacity Project №5: Intro to Machine Learning

## Overview

The goal of this project is to use machine learning (ML) techniques to predict Enron POIs (Persons Of Interest) based on the financial and email information available for Enron employees.
ML is useful here since it allows to process lots of information very quickly compared to human being and reveal dependencies and trends in data. In this project ML will process financial and email information and try to find patterns specific to POIs (whose we know).

Some basic info on the dataset follows:
* The dataset is small and also unbalanced;
* We have total 146 persons on the list;
* Out of them 18 are POIs. The rest (128) are marked as non-POI;
* There are total 23 features in the dataset. One of them is ‘poi’ itself, the feature (label) that we will predict;
* There are 4 (four) features that have a lot of NaN values for POIs. Such features are:
  * director_fees (18 NaN values for POIs);
  * loan_advances (17);
  * restricted_stock_deferred (18);
  * deferral_payments (13);
* There are employees with a lot of missing values. Top 3 include:
  * LOCKHART EUGENE E (20 missing values);
  * GRAMM WENDY L (18);
  * THE TRAVEL AGENCY IN THE PARK (18);

The following outliers were detected and handled:
* THE TRAVEL AGENCY IN THE PARK – this entry does not seem to be an employee. It must be an erroneous entry. The PDF file (enron61702insiderpay.pdf) confirms it. I dropped this entry;
* TOTAL – same as above. I dropped this entry;
* LOCKHART EUGENE E – this person is not a POI and he has all the info missing (all are NaN). I dropped this entry;
* BHATNAGAR SANJAY – this person has several inconsistencies in the financial data. The numbers were manually corrected based on the enron61702insiderpay.pdf file;
* BELFER ROBERT – this person also has several inconsistencies in the financial data. The numbers were manually corrected based on the enron61702insiderpay.pdf file;

Tools for outliers detection included scatter plots, box plots, a sorted list of employees having most NaN values and data re-validation (i.e. total_payments is the sum of several other fields. Same for the stock numbers). 

As another pre-processing step, I replaced all the NaN entries with zeros.

## Feature Selection

Originally, I put all available features - total 20 - into the ‘features_list’ list. Later, after some research, I excluded the following two variables:
* Total_payments
* Total_stock_value

It turned out that algorithms perform better when these are not included.

Then I created two additional features:
* Fraction_to_poi
* Fraction_from_poi

These features are ratios of the number of mails sent to a person from POI to the total number of mails sent to a person, and the number of mails sent from a person to POI to the total number of mails sent from a person, respectively. The rationale is that a real POI might send more mails to other POIs than to other employees. Thus, such ratio could help better identify real POI.

I considered the fact that there might be a possible data leakage with these two.

Feature scaling with MinMaxScaler was used along with the SVM classifier when trying different classifiers.

Using the automated process, which included SelectKBest and StratifiedShuffleSplit, I selected best features to train my classifier on. For SelectKBest I used the default parameter for ‘score_func’ and the other parameter – k, the number of features - was chosen in a loop from the total number of features. One loop iteration follows for illustrative purposes:

Trying k = 12  
Feature Scores:  
exercised_stock_options with the score of 20.286431648  
bonus with the score of 19.1587022727  
salary with the score of 16.5830037466  
fraction_to_poi with the score of 15.0318602558  
deferred_income with the score of 10.6468698012  
long_term_incentive with the score of 9.55380808392  
restricted_stock with the score of 8.41876704524  
shared_receipt_with_poi with the score of 7.92704384697  
loan_advances with the score of 7.31198800461  
expenses with the score of 5.62049485691  
from_poi_to_this_person with the score of 5.44050483697  
other with the score of 5.0588554934  

The best parameters are {'min_samples_split': 5, 'criterion': 'entropy'} with a score of 0.34

Performance of  DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,  
    max_features=None, max_leaf_nodes=None, min_samples_leaf=1,  
    min_samples_split=5, min_weight_fraction_leaf=0.0,  
    presort=False, random_state=None, splitter='best')  

Accuracy: 0.83760	Precision: 0.38203	Recall: 0.35300	
	
Finally, I ended up using just the following two features for my classifier:

1.	bonus with the score of 27.6061992402
2.	exercised_stock_options with the score of 25.9749444917

## Algorithm Selection

My final classifier is Decision Tree with the following parameters:
1.	criterion = ‘gini’;
2.	min_samples_split = 4;
3.	
These parameters were found with the help of GridSearchCV (for parameters selection) and StratifiedShuffleSplit (for cross validation).

I also tried Support Vector Machines, Random Forest and K Nearest Neighbors. My selection steps included:
* Trying all listed classifiers on all features without any feature selection and tuning, just with default parameters;
* Selecting 2 most promising;
* Tuning and selecting the best.

SVM was used in a pipeline after MinMaxScaler.  
All four algorithms were evaluated with StratifiedShuffleSplit.  
SVM returned Precision Rate (P) of 0.29 and Recall Rate (R) of 0.25.  
KM returned P of 0.18 and R of 0.12.   
DT returned P of 0.308 and R of 0.304, and it was a surprise as it is enough to pass the project without any further tuning. Obviously, DT performed best and became the main candidate for a further tuning.  
RF returned P of 0.4 and R of 0.12.  

## Tuning

Most algorithms that I learned have many parameters and each has a default value. In principle, one can use an algorithm with the default values and it will work. This is what I did when trying different algorithms. Tuning the parameters of an algorithm means trying other parameter values other than default ones.

If one does not do it well, one can end up with a severely underperforming algorithm. In other words, one limits the capabilities of ML when not tuning the parameters. On the other hand, properly chosen parameters can noticeably improve the algorithm performance.

In my work, I first selected the features that will be used as an input for my decision tree using SelectKBest and Stratified Shuffle Split. I also tried PCA, but came to a conclusion that SelectKBest provides better results. Then I tried different parameters of DT/RF. I ended up playing with just two parameters: criterion and min_samples_split. In order to find the best value of each parameter I used pipelines, GridSearchCV for parameters selection and StratifiedShuffleSplit for a cross validation.

## Validation

Validation is the process of checking the algorithm performance on an independent set of data. Independent means “independent from the data set that was used to train the algorithm”. If one does it wrong, one can end up with the overfitted model. Often the data set is split into two parts: the train part (used for training) and the test part (used for testing). A good example of a mistake here is to forget to split the data, use all available data for training and then – erroneously – getting high accuracy numbers when trying the algorithm on the same data that used for training.

I validated performance of all my classifiers and models using the StratifiedShuffleSplit cross validation with the number of folds of 1000. The main reason for that is a relatively small amount of targets (POI).

## Evaluation

The two evaluation metrics of my choice are Precision Rate and Recall Rate. When doing the model performance validation based on StratifiedShuffleSplit with 1000 folds, I got the following results:

* Precision: 0.42678
* Recall: 0.40800

In my case of POI, high Precision rate means a high probability of a predicted POI really being a POI. In other words, there is a relatively small number of incorrectly predicted POI. 

High Recall rate means the model did not miss many real POIs when making a prediction. 

