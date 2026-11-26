# Airbnb New User Bookings

Predicting the country where a new Airbnb user will make their first booking, based on the Kaggle competition [Airbnb New User Bookings](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings).

## Problem
Multi-class classification with 12 destination classes. "NDF" (no booking) is the majority class at roughly 58%.

## Data
- train_users_2.csv: about 213k users
- test_users.csv: about 62k users
- sessions.csv: about 10.5M rows of user activity

## Approach
- Exploratory data analysis with seaborn and matplotlib
- Missing values handled with MissForest, KNNImputer and mode imputation
- Outlier clipping on age (IQR)
- Aggregated the 10.5M-row sessions table into per-user features (session count, total seconds, unique action types)
- One-hot encoding
- XGBoost tuned with GridSearchCV (729 fits)

## Results
Best cross-validation accuracy of about 64.2%. The majority class alone gives about 58%, so the gain over a naive baseline is modest.

## Known limitations (v1)
- The imputed first-booking date was used as a feature. This leaks the target, because the date exists only for users who booked.
- Accuracy is not the competition metric. The official metric is NDCG@5.
- Heavy class imbalance means F1 is near zero for most rare countries.
- Age clipping to the IQR range is aggressive.

## Next steps (v2)
Remove the leaking date features, handle class imbalance with sample weights, and evaluate with NDCG@5 and macro-F1.

## Tech
Python, pandas, scikit-learn, XGBoost, seaborn, Google Colab
