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
- Count of unique values for each attribute:
<br>customer_happiness (y) : 54 records (unhappy), 56 records (happy)
<br>delivery_timeliness (X1): 1 (1 records), 2 (0 records), 3 (18 records), 4 (36 records), 5 (55 records)
<br>order_accuracy (X2): 1 (23 records), 2 (30 records), 3 (37 records), 4 (15 records), 5 ( 5 records)
<br>product_availability (X3): 1 (6 records), 2 (13 records), 3 (50 records), 4 (27 records), 5 (14 records)
<br>price_value_perception (X4): 1 (2 records), 2 (5 records), 3 (35 records), 4 (46 records), 5 (22 records) 
<br>courier_service_rating (X5): 1 (7 records), 2 (16 records), 3 (19 records), 4 (41 records), 5 (27 records)
<br>app_usability (X6): 1 (1 records), 2 (1 records), 3 (20 records), 4 (40 records), 5 (48 records)

> Target attribute (Y) is balanced with almost equal number of records for both classes. All other attributes have values from 1 to 5 with varying distribution.


> A stratified train-test split was used to preserve the original class distribution of happy and unhappy customers in both training and testing datasets. This reduces sampling bias and provides a more reliable estimate of model performance, particularly given the small dataset size (110 observations).

## Models Evaluated
1] Logistic Regression
<br>2] Random Forest
<br>3] XGBoost 
<br>4] K-Nearest Neighbors (KNN) 
<br>5] Support Vector Machine (SVM)
<br>6] Naive Bayes

### Model Comparison Summary
Random Forest: Accuracy = 63.64%, ROC-AUC = 0.6198
<br>KNN: Accuracy = 63.64%, ROC-AUC = 0.6529
<br>SVM: Accuracy = 59.09%, ROC-AUC = 0.6529
<br>Logistic Regression: Accuracy = 54.55%, ROC-AUC = 0.6446
<br>Naive Bayes: Accuracy = 54.55%, ROC-AUC = 0.6364
<br>XGBoost: Accuracy = 50.00%, ROC-AUC = 0.5620

### Overall Feature Ranking Across Models
**delivery_timeliness** was ranked as the **most** important feature overall, with an average rank of 2.50.
<br>**courier_service_rating** was the **second** most important feature, with an average rank of 2.83.
<br>**product_availability** ranked **third** overall, with an average rank of 3.00.
<br>**app_usability** was ranked **fourth**, with an average rank of 3.67.
<br>**price_value_perception** ranked **fifth**, with an average rank of 3.83.
<br>**order_accuracy** was the **least** important feature overall, with an average rank of 4.50.

## SHAP Feature Importance Analysis

### Most Important Feature by Model (SHAP Analysis):
**Logistic Regression** identified **delivery_timeliness** as the most important feature (SHAP importance = 0.3901).
<br>**Random Forest** identified **delivery_timeliness** as the most important feature (SHAP importance = 0.0921).
<br>**KNN** identified **delivery_timeliness** as the most important feature (SHAP importance = 0.0668).
<br>**SVM** identified **delivery_timeliness** as the most important feature (SHAP importance = 0.0001).

### Least Important Feature by Model (SHAP Analysis):
**Logistic Regression** identified **app_usability** as the least important feature (SHAP importance = 0.0015).
<br>**Random Forest** identified **order_accuracy** as the least important feature (SHAP importance = 0.0307).
<br>**KNN** identified **app_usability** as the least important feature (SHAP importance = 0.0294).
<br>**SVM** identified **order_accuracy** as the least important feature (SHAP importance = 0.0000).

### Minimum Feature Set Achieving Maximum Accuracy
Logistic Regression:
- Accuracy: 59.09%
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating

Random Forest:
- Accuracy: 63.64%
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating
  - app_usability
  - order_accuracy

KNN:
- Accuracy: 63.64%
- Features:
  - delivery_timeliness
  - product_availability
  - price_value_perception
  - courier_service_rating
  - app_usability
  - order_accuracy

SVM:
- Accuracy: 59.09%
- Features:
  - delivery_timeliness
  - product_availability
  
## Hyperparameter tuning the models.
SVM achieved the highest cross-validation accuracy (56.73%).
<br>Logistic Regression and Random Forest performed similarly (54.58%).
<br>KNN achieved the lowest cross-validation accuracy (53.33%).

> After hyperparameter tuning, the best accuracy achieved was 0.567320 with SVM model. The most important feature for SVM was found to be 'product_availability' and the least important feature was 'order_accuracy'. 

> An accurracy of 73% or above was not achieved with any of the models evaluated. This could be because the target is extremely balanced and the dataset is small.

## Tune the models with the best hyperparameters for the minimum feature set to get best accuracy.

Since SVM had the most accuracy with the hyperparameter.
<br>With best hyperparameters: {'C': 0.01, 'gamma': 'scale', 'kernel': 'linear'}
<br>and minimum feature set ['delivery_timeliness', 'product_availability']
 - Accuracy: 0.5
 - ROC-AUC: 0.39256198347107435

Tuned Random Forest:
<br>with best hyperparameters: {'max_depth': 2, 'min_samples_leaf': 4, 'n_estimators': 50}
<br>and minimum feature set is all the features.
 - Accuracy: 0.5454545454545454
 - ROC-AUC: 0.6115702479338843

Tune Logistic Regression:
<br>with best hyperparameters: {'C': 0.001, 'penalty': 'l2', 'solver': 'liblinear'}
<br>and minimum feature set is ['delivery_timeliness', 'product_availability', 'price_value_perception', 'courier_service_rating'].
 - Accuracy: 0.5
 - ROC-AUC: 0.6652892561983471

Tuned KNN:
<br>with best hyperparameters: {'n_neighbors': 7, 'weights': 'uniform'}
<br>and minimum feature set is all the features.
 - Accuracy: 0.6818181818181818
 - ROC-AUC: 0.7892561983471075