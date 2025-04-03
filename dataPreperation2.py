import pandas as pd
import nltk
import re
from string import punctuation
import text_mining_utils as tmu

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns




# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop extra columns and rename
data = data.iloc[:, :2]
data.columns = ["label", "text"]

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

documents = data['text'].tolist()

count_matrix = tmu.build_count_matrix(data["text"])
tf_matrix = tmu.build_tf_matrix(data["text"])
tfidf_matrix = tmu.build_tfidf_matrix(data["text"])

# print matrices
print(count_matrix)
print(tf_matrix)
print(tfidf_matrix)

from sklearn.feature_selection import chi2, f_classif

# Apply univariate feature selection using Chi-Square
features_scores_chi2, X_reduced_chi2 = tmu.stat_univariate_fs(
    tfidf_matrix,  
    data["label"],  
    weight_method=chi2,  
    selection_method='percentile', 
    param=10  
)

# Print the top features selected by Chi-Square
print("Top Features (Chi-Square):")
print(features_scores_chi2.sort_values(by="Weight", ascending=False).head(10))

# Apply univariate feature selection using ANOVA F-value
features_scores_f, X_reduced_f = tmu.stat_univariate_fs(
    tfidf_matrix,  
    data["label"],  
    weight_method=f_classif,  
    selection_method='percentile',  
    param=10 
)

# Print the top features selected by ANOVA F-value
print("Top Features (ANOVA F-value):")
print(features_scores_f.sort_values(by="Weight", ascending=False).head(10))

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced_f , data["label"], test_size=0.2, random_state=42)

# Train a classifier
clf = SVC(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

