# Income-Classification-Using-Machine-Learning-on-Census-Data
This project focuses on predicting whether a person's income exceeds $50K per year using the Census Income dataset (Adult Data Set) from the UCI Machine Learning Repository. The problem is framed as a binary classification task, and multiple machine learning models have been implemented, compared, and evaluated to achieve accurate predictions.
Objective:
       1]Classify individuals based on demographic and employment-related features.
       2]Explore and compare the performance of various supervised learning algorithms.
       3]Apply proper preprocessing and hyperparameter tuning techniques to enhance model performance.
The dataset underwent thorough preprocessing before modeling:
       1]Handling Missing Values: Replaced or dropped records with missing or unknown values.
       2]Label Encoding: Converted categorical variables to numeric form using LabelEncoder.
       3]Outlier Treatment: Identified and handled outliers in numerical features (e.g., age, hours-per-week) using visual inspection and IQR method.
       4]Feature Engineering: Removed or transformed skewed or less relevant features to improve model generalization.
Machine Learning Models Used:
       1]Logistic Regression
       2]Decision Tree Classifier
       3]Random Forest Classifier
       4]Hyperparameter Tuning with GridSearchCV on the Random Forest model to find the optimal combination of:
            1)n_estimators
            2)max_depth
            3)min_samples_split
Evaluation Metrics:
Models were evaluated using accuracy_score
