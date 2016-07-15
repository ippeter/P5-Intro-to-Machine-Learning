#!/usr/bin/python

import sys
import pickle
import operator
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import defaultdict

from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.preprocessing import MinMaxScaler

# General steps:
seed = 7
np.random.seed(seed)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'deferred_income', 'expenses', 'other', 'long_term_incentive', \
                'director_fees', 'loan_advances', 'deferral_payments', \
                'restricted_stock', 'exercised_stock_options', 'restricted_stock_deferred', \
                'from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers and fix wrong entries
# First some preprocessing: replace all NaN with 0
for emp in data_dict:
    for k in data_dict[emp]:
        if (data_dict[emp][k] == 'NaN'):
            data_dict[emp][k] = 0

# Second outliers that I want to remove
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

# Finally two items that have to be fixed manually
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 0

data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def computeFraction( poi_messages, all_messages ):
    """ 
    This function is stolen from the class as is!
    Given a number messages to/from POI (numerator) and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person that are from/to a POI
   """
    if (poi_messages == 0) or (all_messages == 0):
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction

for name in my_dataset:
    data_point = my_dataset[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )

    my_dataset[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    my_dataset[name]["fraction_to_poi"] = fraction_to_poi

# Add the new features to the features list    
features_list.append("fraction_from_poi")
features_list.append("fraction_to_poi")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Implement stratified shuffle split
def do_sss(c, l, f):
    """
    This function does the stratified shuffle split for my predicted data
    Input: c - classifier, l - labels, f - features
    Output: None. Prints accuracy, precision rate, recall rate
    """
    PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\t"
    
    sss = StratifiedShuffleSplit(l, n_iter = 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    
    for train_idx, test_idx in sss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( f[ii] )
            labels_train.append( l[ii] )
        for jj in test_idx:
            features_test.append( f[jj] )
            labels_test.append( l[jj] )
        
        ### fit the classifier using training set, and test on test set
        c.fit(features_train, labels_train)
        predictions = c.predict(features_test)
        
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)

        print "Performance of ", c
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, display_precision = 5)
        print
    except:
        print "Got a divide by zero when trying out:", c
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        print
    return None

# I decided to start with the SVM classifier instead of NB
clf1 = Pipeline(steps=[ \
                    ('scaler', MinMaxScaler()), \
                    ('classifier', SVC(C=10000.0, kernel = 'rbf', gamma = 'auto', degree = 3)) \
                    ])
do_sss(clf1, labels, features)

# Let's see how Decision Tree does
clf2 = tree.DecisionTreeClassifier()
do_sss(clf2, labels, features)

# Let's try Random Forest
clf3 = RandomForestClassifier()
do_sss(clf3, labels, features)

# Finally goes K-Means
clf4 = cluster.KMeans(n_clusters = 2)
do_sss(clf4, labels, features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Better way of Feature Selection as suggested in the first review
# Dear reviewer, I didn't understand how to automate the feature selection process better than this :(
# I loop through all possible values and then check each value with SSS, then apply DT to the result.
# It will take a few minutes, please be patient!
# There will be a lot of output, but one will clearly see which features perform best for DT.
for k in range(1, len(features_list) - 1):
    print "Trying k =", k
    
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # These Numpy arrays will be used to store intermediate results
    np_scores = np.zeros(len(features_list) - 1)
    np_occurs = np.zeros(len(features_list) - 1)

    sss_cv_splits = StratifiedShuffleSplit(labels, n_iter = 1000, random_state = 42)

    for i_train, i_test in sss_cv_splits:
        features_train, features_test = [features[i] for i in i_train], [features[i] for i in i_test]
        labels_train, labels_test = [labels[i] for i in i_train], [labels[i] for i in i_test]

    # fit selector to training set
        sel = SelectKBest(f_classif, k = k)
        sel.fit(features_train, labels_train)

        # Store scores and features in Numpy arrays for fast further processing
        np_scores = np_scores + np.array(sel.scores_)
        np_occurs = np_occurs + np.array(sel.get_support())

    # Normalize scores and convert to list of tuples for further sorting
    # This is to avoid all zero-occurances
    func = np.vectorize(lambda x: 1 if (x == 0) else x)
    np_occurs = func(np_occurs)
    
    # Normalize scores
    np_scores = np_scores / np_occurs
    
    # Find indexes of k most commonly occurred features
    best_f_indices = np_occurs.argsort()[-k:]
    
    # Make a Numpy array with final feature names and their scores
    np_final_features_list = np.array(features_list[1:])[best_f_indices]
    np_final_scores = np_scores[best_f_indices]
    
    # Convert a Numpy feature list to a python list
    final_features_list = ['poi'] + np_final_features_list.tolist()
    
    # Finally print feature names and their average scores
    print "Feature Scores:"
    for i in np_final_scores.argsort()[::-1]:
        print final_features_list[i + 1], "with the score of", np_final_scores[i]
    print
        
    # Reshape our data using final_features_list, the final list of features
    data = featureFormat(my_dataset, final_features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # Now tune the DT algorith using the GridSearchCV
    # We then store the split instance into cv and use it in our GridSearchCV.
    sss = StratifiedShuffleSplit(labels, n_iter = 1000, random_state = 42)
    clf = tree.DecisionTreeClassifier()
    parameters = {'min_samples_split':[1, 2, 3, 4, 5, 7, 8, 9], \
              'criterion':['gini', 'entropy']} 

    grid = GridSearchCV(clf, parameters, cv = sss, scoring='f1')
    grid.fit(features, labels)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    clf = tree.DecisionTreeClassifier(criterion = grid.best_params_['criterion'], \
                                  min_samples_split = grid.best_params_['min_samples_split'])
    do_sss(clf, labels, features)
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Based on the above, I see that the best result is achieved with just 2 best features
# So here is the final feature list:
final_features_list = ['poi', 'bonus', 'exercised_stock_options']

# Also, I noticed that the best performance was achieved with 'gini' and min_samples_split of 4:
clf = tree.DecisionTreeClassifier(criterion = 'gini', min_samples_split = 4)   

# Thus I export my classifier as follows:
dump_classifier_and_data(clf, my_dataset, final_features_list)
