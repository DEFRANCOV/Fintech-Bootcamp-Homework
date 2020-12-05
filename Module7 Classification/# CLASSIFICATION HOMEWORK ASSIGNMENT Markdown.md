# CLASSIFICATION HOMEWORK ASSIGNMENT

## Credit Risk Resampling

### Reading & Cleaning
- After we read in the CSV file, we check for nulls and duplicates.
- There were no nulls.   There were some duplicates but as there is no unique identifier that is duplicated, we can’t eliminate the duplicates.
- We also import the LabelEncoder from sklearn.preprocessing.   This function will enable us to transform the string values in the homeowner column to 0s, 1s and 2s denoting mortgage, ownership, and renter.

### Split the Data
- We create two dataframes.  The first one is for X which drops “loan status” column from the original dataframe.   y is the dependent variable so it’s a single column dataframe of loan status.
- We then use model_selection from sklearn to split X and y into training and test dataframes.  

### Simple Logistic Regression
- From sklearn, we utilize the Logistic Regression function.
- The balanced accuracy score is 0.95432 and the recall for the high risk is 0.9136.

### Oversampling:  Random Oversampling
- From imblearn, we import RandomOverSampler.
- We resample (“oversample”) the X_train and y_train data through RandomOverSampler.   The sample is 56277 High Risk and Low Risk.
- We run a logistic regression that was formed on fitting X-resampled and y_resampled. 
- In this case, the balanced accuracy score is 0.9948 and the recall for the high risk is 0.9952.    


### Oversampling:  SMOTE
- From imblearn, we import SMOTE.
- We resample (“oversample”) X_train and y_train data through SMOTE.   The sample is 56277 for each High Risk and Low Risk.
- We run a logistic regression that was formed on fitting X-resampled2 and y_resampled2. 
- In this case, the balanced accuracy score is 0.99483 and the recall for the high risk is 0.9952.    


### Undersampling
- From imblearn, we import ClusterCentroids.
- We resample (“undersample”) X_train and y_train data through ClusterCentroids.   The sample is 1875 for each High Risk and Low Risk.
- We run a logistic regression that was formed on fitting X-resampled3 and y_resampled3. 
- In this case, the balanced accuracy score is 0.98368 and the recall for the high risk is 0.9728.    


### Combination sampling
- From imblearn, we import SMOTEENN.
- We resample (“over and undersample”) X_train and y_train data through SMOTEENN.   The sample is 74117 and 74577 for High Risk and Low Risk.  
- We run a logistic regression that was formed on fitting X-resampled4 and y_resampled4. 
- In this case, the balanced accuracy score is 0.99475 and the recall for the high risk is 0.9952.    

### Conclusion
- Overall, the SMOTE Oversampling produces the best results although Random Oversampling and SMOTEENN are a close 2nd and 3rd place.  
- For the recall-High Risk, all three have the same score.   
- However, SMOTE has the highest balanced accuracy score and the best F1 score for high risk.




## Credit Risk Ensemble

### Reading & Cleaning
- The same steps are performed here as in Credit Resampling.
 
### Split the Data
- The same steps are performed here as in Credit Resampling.
- Scaling the data does not make any difference in outcomes.

### Balanced Random Forest Classifier
- From imblearn, we import BalancedRandomForestClassifier.  
- We use BalancedRandomForestClassifier to fix X_train and y_train.
- We run this model on X_test and come up with y_pred_rf.  
- In this case, the balanced accuracy score is 0.99374 and the recall for the high risk is 0.9952.    
- Finally, we use the feature_importances_ function that is part of BalancedRandomForestClassifier.   The top three features are “total debt”, “borrower income”, and “debt to income”.  “debt to income” is not independent of “total debt” and “borrower income”.  

### Easy Ensemble Classifier
- From imblearn, we import EasyEnsembleClassifier.  
- We use EasyEnsembleClassifier to fix X_train and y_train.
- We run this model on X_test and come up with y_pred_eec.  
- In this case, the balanced accuracy score is 0.99445 and the recall for the high risk is 0.9952.    

### Conclusion
- The Easy Ensemble model slightly beats out BalancedRandomForest.   They have equivalent scores on recall-high risk but Easy Ensemble has a better balanced accuracy score and a better F1 score on high risk.  
