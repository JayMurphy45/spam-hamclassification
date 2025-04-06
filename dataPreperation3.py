import pandas as pd
import nltk
import re
from string import punctuation
import text_mining_utils as tmu

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Build vectorization matrices
count_matrix = tmu.build_count_matrix(data["text"])
tf_matrix = tmu.build_tf_matrix(data["text"])
tfidf_matrix = tmu.build_tfidf_matrix(data["text"])

# Print matrices (optional)
print(count_matrix)
print(tf_matrix)
print(tfidf_matrix)

# Apply Sequential Feature Selection using SVC with linear kernel
svc = SVC(kernel='linear', random_state=42)

# Use SFS to reduce TF-IDF matrix features
X_reduced_sfs = tmu.sequential_fs(
    tfidf_matrix, 
    data["label"], 
    learner=svc, 
    num_features=50,     
    direction='forward', 
    cv=5, 
    scoring='f1_macro'
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced_sfs, data["label"], test_size=0.2, random_state=42
)

# Train classifier
clf = SVC(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))