# Customer Happiness Prediction

## Problem Statement
1] Predict if a customer is happy or not based on the answers they give to questions asked.
   -> Reach 73% accuracy score or above.
2] Find which features are more important when predicting a customer’s happiness.
  -> Minimal set of features that increases predictability of the data.
  -> Feature of least importance that can be removed.

## Dataset
Y (customer_happiness) = target attribute (Y) with values indicating 0 (unhappy) and 1 (happy) customers
X1 (delivery_timeliness) = my order was delivered on time
X2 (order_accuracy) = contents of my order was as I expected
X3 (product_availability) = I ordered everything I wanted to order
X4 (price_value_perception) = I paid a good price for my order
X5 (courier_satisfaction) = I am satisfied with my courier
X6 (app_usability) = the app makes ordering easy for me 

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5 where the smaller number indicates less and the higher number indicates more towards the answer.

> Supervised Learning - as we already know the correct answer (happy\unhappy).
We have features (delivery time lines, order accuracy, product availability, price value perception, courier service rating, app usability) to help predict target (happy\unhappy).
> It is Binary Classification as there are exactly two possible outcomes - happy or unhappy.

## Exploratory Data Analysis
- Dataset contains 126 records.
- List all unique values for each attribute.
- Check for missing values in the dataset - no null values found.
- Check for duplicates in the dataset - 16 duplicate records found.
- Count of unique values for each attribute:
    customer_happiness (y) : 54 records (unhappy),
                             56 records (happy)
    delivery_timeliness (X1): 1 - 1 records, 
                              2 - 0 records, 
                              3 - 18 records, 
                              4 - 36 records, 
                              5 - 55 records
    order_accuracy (X2): 1 - 23 records, 
                         2 - 30 records, 
                         3 - 37 records, 
                         4 - 15 records, 
                         5 - 5 records
    product_availability (X3): 1 - 6 records, 
                               2 - 13 records, 
                               3 - 50 records, 
                               4 - 27 records, 
                               5 - 14 records
    price_value_perception (X4): 1 - 2 records, 
                                 2 - 5 records, 
                                 3 - 35 records, 
                                 4 - 46 records,
                                 5 - 22 records 
    courier_service_rating (X5): 1 - 7 records, 
                                 2 - 16 records, 
                                 3 - 19 records, 
                                 4 - 41 records, 
                                 5 - 27 records
    app_usability (X6): 1 - 1 records, 
                            2 - 1 records, 
                            3 - 20 records, 
                            4 - 40 records, 
                            5 - 48 records

> Target attribute (Y) is balanced with almost equal number of records for both classes. All other attributes have values from 1 to 5 with varying distribution.


> A stratified train-test split was used to preserve the original class distribution of happy and unhappy customers in both training and testing datasets. This reduces sampling bias and provides a more reliable estimate of model performance, particularly given the small dataset size (110 observations).

## Models Evaluated
1] Logistic Regression
2] Random Forest
3] XGBoost 
4] K-Nearest Neighbors (KNN) 
5] Support Vector Machine (SVM)
6] Naive Bayes

MODEL COMPARISON SUMMARY
                     Accuracy   ROC_AUC
Model                                  
Logistic Regression  0.545455  0.644628
Random Forest        0.636364  0.619835
XGBoost              0.500000  0.561983
KNN                  0.636364  0.652893
SVM                  0.590909  0.652893
Naive Bayes          0.545455  0.636364

Feature	                Rank_LR	Rank_RF	Rank_XGB	Rank_KNN	Rank_SVM	Rank_NB	Average_Rank
delivery_timeliness	      1	       5	     1	       2	      4	          2	    2.500000
courier_service_rating	  3	       3	     5	       3	      2	          1	    2.833333
product_availability	    4	       1	     4	       5	      1	          3	    3.000000
app_usability	            6	       2	     3	       4	      3	          4	    3.666667
price_value_perception	  2	       6	     2	       2	      5	          6	    3.833333
order_accuracy	          5	       4	     6	       1	      6	          5	    4.500000

## SHAP Feature Importance Analysis

### Most Important Feature by Model (SHAP Analysis):
Model                Most Important Feature  SHAP Importance 
Logistic Regression  delivery_timeliness      0.390131
Random Forest        delivery_timeliness      0.092140
KNN                  delivery_timeliness      0.066770
SVM                  delivery_timeliness      0.000099

### Least Important Feature by Model (SHAP Analysis):
Model                 Least Important Feature  SHAP Importance
Logistic Regression   app_usability            0.001547
Random Forest         order_accuracy           0.030677
KNN                   app_usability            0.029359
SVM                   order_accuracy           0.000025

### Minimum Feature Set Achieving Maximum Accuracy
Logistic Regression
- Accuracy: **59.09%**
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating

Random Forest
- Accuracy: **63.64%**
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating
  - app_usability
  - order_accuracy

KNN
- Accuracy: **63.64%**
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating
  - app_usability
  - order_accuracy

SVM
- Accuracy: **59.09%**
- Features:
  - delivery_timeliness
  - product_availability
  
## Hyperparameter tuning the models.
              Model  Best_CV_Accuracy       
Logistic Regression          0.545752
      Random Forest          0.545752
                KNN          0.533333
                SVM          0.567320

> After hyperparameter tuning, the best accuracy achieved was 0.567320 with SVM model. The most important feature for SVM was found to be 'product_availability' and the least important feature was 'order_accuracy'. 

> An accurracy of 73% or above was not achieved with any of the models evaluated. This could be because the target is extremely balanced and the dataset is small.

## Tune the models with the best hyperparameters for the minimum feature set to get best accuracy.

Since SVM had the most accuracy with the hyperparameter.
With best hyperparameters: {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}
and minimum feature set ['delivery_timeliness', 'product_availability']
Accuracy: 0.5
ROC-AUC: 0.39256198347107435

Tuned Random Forest:
with best hyperparameters: {'max_depth': 2, 'min_samples_leaf': 4, 'n_estimators': 50}
and minimum feature set is all the features.
Accuracy: 0.5454545454545454
ROC-AUC: 0.6115702479338843

Tune Logistic Regression:
with best hyperparameters: {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
and minimum feature set is ['delivery_timeliness',
                            'product_availability',
                            'price_value_perception',
                            'courier_service_rating'].
Accuracy: 0.5
ROC-AUC: 0.6652892561983471

Tuned KNN:
with best hyperparameters: {'n_neighbors': 7, 'weights': 'uniform'}
and minimum feature set is all the features.
Accuracy: 0.6818181818181818
ROC-AUC: 0.7892561983471075