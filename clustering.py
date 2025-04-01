from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import text_mining_utils as tmu

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin1")

# Drop extra columns and rename
data = data.iloc[:, :2]
data.columns = ["label", "text"]

data['label_num'] = data['label'].map({'spam': 1, 'ham': 0})

# Step 1: Vectorize text using TF-IDF
X_tfidf = tmu.build_tfidf_matrix(data['text'])

# Step 2: Apply K-Means clustering
k = 2
model, cluster_labels, cluster_centers = tmu.k_means_clustering(X_tfidf, k, initialisation='k-means++')

# Step 3: Assign clusters to actual Spam/Ham labels
true_labels = data['label_num'].values  # Ground truth labels
cluster_mapping = {}

# Determine majority class per cluster
for cluster in range(k):
    majority_class = np.bincount(true_labels[cluster_labels == cluster]).argmax()
    cluster_mapping[cluster] = majority_class

# Map cluster predictions to Spam/Ham labels
predicted_labels = np.array([cluster_mapping[label] for label in cluster_labels])

# Step 4: Evaluate clustering accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"K-Means Clustering Accuracy: {accuracy:.2f}")

# Step 5: Visualize cluster centroids
tmu.centroids_across_terms(X_tfidf, cluster_centers, cluster_labels, "Cluster Centroids")
plt.show()
