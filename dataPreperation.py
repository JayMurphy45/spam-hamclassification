import pandas as pd
import nltk
import re
from string import punctuation
import text_mining_utils as tmu

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
from nltk.corpus import stopwords

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop extra columns and rename
data = data.iloc[:, :2]
data.columns = ["label", "text"]

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Define a custom stopword list
custom_stopwords = set(stopwords.words("english"))
custom_stopwords.update(["you", "to", "a", ",", "."]) 

# Function to remove stopwords
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)  # Tokenize text
    cleaned_text = " ".join([word for word in tokens if word.lower() not in custom_stopwords and word not in punctuation])
    return cleaned_text

# Apply stopword removal
data["cleaned_text"] = data["text"].apply(remove_stopwords)

print(data["cleaned_text"].isnull().sum())
print((data["cleaned_text"] == "").sum())
data = data[data["cleaned_text"].str.strip() != ""]
print((data["cleaned_text"] == "").sum())

# Show the difference
print(data[["text", "cleaned_text"]].head())

# build matrices
count_matrix = tmu.build_count_matrix(data["cleaned_text"])
print(count_matrix)

tf_matrix = tmu.build_tf_matrix(data["cleaned_text"])
print(tf_matrix)

tfidf_matrix = tmu.build_tfidf_matrix(data["cleaned_text"])
print(tfidf_matrix)



# ------------------------------------------------------------------------
# Extract target labels
y = data["label"]

# Split dataset into training (80%) and testing (20%)
X_train_count, X_test_count, y_train, y_test = train_test_split(
    count_matrix, y, test_size=0.2, random_state=42
)
X_train_tf, X_test_tf, _, _ = train_test_split(
    tf_matrix, y, test_size=0.2, random_state=42
)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(
    tfidf_matrix, y, test_size=0.2, random_state=42
)

# Initialize the support vector classifier
clf = SVC(random_state=1)

# Evaluate model performance on training data
tmu.print_classif_report(clf, X_train_count, y_train)
tmu.print_classif_report(clf, X_train_tf, y_train)
tmu.print_classif_report(clf, X_train_tfidf, y_train)