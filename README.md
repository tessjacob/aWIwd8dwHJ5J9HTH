# Customer Happiness Prediction

## Problem Statement
1] Predict if a customer is happy or not based on the answers they give to questions asked.
   <br>-> Reach 73% accuracy score or above.
<br>2] Find which features are more important when predicting a customer’s happiness.
  <br>-> Minimal set of features that increases predictability of the data.
  <br>-> Feature of least importance that can be removed.

## Dataset
Y (customer_happiness) = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
<br>X1 (delivery_timeliness) = my order was delivered on time
<br>X2 (order_accuracy) = contents of my order was as I expected
<br>X3 (product_availability) = I ordered everything I wanted to order
<br>X4 (price_value_perception) = I paid a good price for my order
<br>X5 (courier_satisfaction) = I am satisfied with my courier
<br>X6 (app_usability) = the app makes ordering easy for me 

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.

> Supervised Learning - as we already know the correct answer (happy\unhappy).
We have features (delivery time lines, order accuracy, product availability, price value perception, courier service rating, app usability) to help predict target (happy\unhappy).
> It is Binary Classification as there are exactly two possible outcomes - happy or unhappy.

## Exploratory Data Analysis
- Dataset contains 126 records.
- List all unique values for each attribute.
- Check for missing values in the dataset - no null values found.
- Check for duplicates in the dataset - 16 duplicate records found.
<br>* Duplicate rows are not deleted.
- Count of unique values for each attribute:
<br>customer_happiness (y) : 54 records (unhappy), 56 records (happy)
<br>delivery_timeliness (X1): 1 (1 records), 2 (0 records), 3 (18 records), 4 (36 records), 5 (55 records)
<br>order_accuracy (X2): 1 (23 records), 2 (30 records), 3 (37 records), 4 (15 records), 5 ( 5 records)
<br>product_availability (X3): 1 (6 records), 2 (13 records), 3 (50 records), 4 (27 records), 5 (14 records)
<br>price_value_perception (X4): 1 (2 records), 2 (5 records), 3 (35 records), 4 (46 records), 5 (22 records) 
<br>courier_service_rating (X5): 1 (7 records), 2 (16 records), 3 (19 records), 4 (41 records), 5 (27 records)
<br>app_usability (X6): 1 (1 records), 2 (1 records), 3 (20 records), 4 (40 records), 5 (48 records)

> Target attribute (Y) is balanced with almost equal number of records for both classes. All other attributes have values from 1 to 5 with varying distribution.

> A stratified train-test split was used to preserve the original class distribution of happy and unhappy customers in both training and testing datasets. This reduces sampling bias and provides a more reliable estimate of model performance, particularly given the small dataset size (126 observations).

## Models Evaluated
1] Logistic Regression
<br>2] Random Forest
<br>3] XGBoost 
<br>4] K-Nearest Neighbors (KNN) 
<br>5] Support Vector Machine (SVM)
<br>6] Naive Bayes

### Model Comparison Summary
Random Forest: Accuracy = 73.07%, ROC-AUC = 0.7976
<br>KNN: Accuracy = 61.53%, ROC-AUC = 0.5744
<br>SVM: Accuracy = 61.53%, ROC-AUC =  0.7440
<br>Logistic Regression: Accuracy = 61.53%, ROC-AUC = 0.7500
<br>Naive Bayes: Accuracy = 61.53%, ROC-AUC = 0.76785
<br>XGBoost: Accuracy = 61.53%, ROC-AUC = 0.7261

> Random Forest and KNN have most accuracy followed by SVM and Logistic Regression.

### Overall Feature Ranking Across Models
delivery_timeliness (Average Rank: 3.333333)
<br>product_availability (Average Rank: 2.500000)
<br>price_value_perception (Average Rank: 2.666667)
<br>order_accuracy (Average Rank: 3.166667)
<br>app_usability (Average Rank: 4.000000)
<br>courier_service_rating (Average Rank: 3.000000)

> delivery_timeliness and product_availability are the most important features across all models.
> courier_service_rating is the least important feature.

## SHAP Feature Importance Analysis

### Most Important Feature by Model (SHAP Analysis):
Logistic Regression: delivery_timeliness (0.2715)
<br>Random Forest: delivery_timeliness (0.1108)
<br>KNN: order_accuracy (0.0412)
<br>SVM: delivery_timeliness (0.0004)

> delivery_timeliness was identified as the most important feature by 3 out of 4 models.
<br>> order_accuracy was considered most important only by the KNN model.

### Least Important Feature by Model (SHAP Analysis):
Logistic Regression: price_value_perception (0.0046)
<br>Random Forest: app_usability (0.0074)
<br>KNN: delivery_timeliness (1.6931)
<br>SVM: order_accuracy (0.0001)

> price_value_perception was the least influential feature or Logistic Regression.
<br>> app_usability was the least influential feature for Random Forest.
<br>> delivery_timeliness was the least influential feature for KNN.
<br>> order_accuracy had the least impact in the SVM.

### Minimum Feature Set Achieving Maximum Accuracy
Logistic Regression:
  - Accuracy: 0.6538461538461539
  - Features:
    - delivery_timeliness
    - product_availability
    - price_value_perception

Random Forest:
  - Accuracy: 0.6538461538461539
  - Features:  
    - delivery_timeliness
    - product_availability
    - price_value_perception
    - courier_service_rating

KNN:
  - Accuracy: 0.6538461538461539
  - Features:
    - delivery_timeliness
    - product_availability

SVM:
  - Accuracy: 0.6923076923076923
  - Features:
    - delivery_timeliness
    - product_availability
    - price_value_perception
    - courier_service_rating
    - app_usability
  
## Hyperparameter tuning the models.
SVM: Accuracy = 56%, Standard Deviation = 0.0734
<br>Logistic Regression: Accuracy = 54%, Standard Deviation = 0.1157
<br>Random Forest: Accuracy = 57%, Standard Deviation = 0.1029
<br>KNN: Accuracy = 57%, Standard Deviation = 0.06

## Tune the models with the best hyperparameters for the minimum feature set to get best accuracy.

### Tuned SVM.
<br>With best hyperparameters: {'svm__C': 0.1, 'svm__gamma': 'scale', 'svm__kernel': 'linear'}
<br>and minimum feature set ['delivery_timeliness',
    'product_availability',
    'price_value_perception',
    'courier_service_rating',
    'app_usability']
<br> and ['delivery_timeliness', 
'product_availability', 
'price_value_perception', 
'courier_service_rating', 
'app_usability', 'order_accuracy']
 - Accuracy: 0.7916666666666666

### Tuned Random Forest:
<br>with best hyperparameters: {'rf__max_depth': 4, 'rf__min_samples_leaf': 1, 'rf__n_estimators': 50}
<br>and minimum feature set is ['delivery_timeliness',
    'product_availability',
    'price_value_perception',
    'courier_service_rating']
<br> and ['delivery_timeliness', 
    'product_availability', 
    'price_value_perception', 
    'courier_service_rating', 
    'app_usability', 'order_accuracy'].
 - Accuracy: 0.8214285714285714

### Tune Logistic Regression:
<br>with best hyperparameters: {'lr__C': 0.001, 'lr__penalty': 'l2', 'lr__solver': 'liblinear'}
<br>and minimum feature set is ['delivery_timeliness',
    'product_availability',
    'price_value_perception'].
 - Accuracy: 0.6538461538461539

### Tuned KNN:
<br>with best hyperparameters: {'knn__metric': 'euclidean', 'knn__n_neighbors': 5, 'knn__weights': 'distance'}
<br>and minimum feature set ['delivery_timeliness',
      'product_availability']
<br> and ['delivery_timeliness', 'product_availability', 'price_value_perception'].
 - Accuracy: 0.5476190476190476

> After tuning the models with the hyperparameter and minimum feature set KNN can achive the accuracy of 68%.