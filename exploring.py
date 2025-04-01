import pandas as pd
import numpy as np
import nltk
import re
from string import punctuation
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import text_mining_utils as tmu

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnamed extra columns
data = data.iloc[:, :2]

# Rename columns
data.columns = ["label", "text"]

# Check for missing values
print(data.isnull().sum())

# Show dataset info
data.info()

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Split texts into spam and ham categories (convert to lists)
spam_texts = data[data["label"] == 1]["text"].tolist()
ham_texts = data[data["label"] == 0]["text"].tolist()

# 1. Find 10 most frequent words in each category
spam_text_combined = " ".join(spam_texts)
ham_text_combined = " ".join(ham_texts)

print("Spam Category:")
tmu.print_n_most_frequent("Spam", spam_text_combined, 10)

print("\nHam Category:")
tmu.print_n_most_frequent("Ham", ham_text_combined, 10)

# 2. Compute token percentage of specific words (e.g., "free", "win", "offer")
for word in ["free", "win", "offer", "FREE", "WIN", "OFFER", "!", "?","call"]:
    spam_percentage = tmu.token_percentage(word, spam_texts)
    ham_percentage = tmu.token_percentage(word, ham_texts)
    
    print(f"\nToken '{word}' percentage in Spam: {spam_percentage:.2f}%")


# Prepare the text and categories list
texts = [spam_text_combined, ham_text_combined]
categories = ["Spam", "Ham"]

# Generate word clouds
# tmu.generate_wordclouds(texts, categories, bg_colour='white')


# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

def count_capitalized_words(text):
    return len(re.findall(r'[A-Z]{2,}', text))

# Apply function to each message
data["capital_count"] = data["text"].apply(count_capitalized_words)

# Calculate average capitalized words per message for spam and ham
spam_avg = data[data["label"] == 1]["capital_count"].mean()
ham_avg = data[data["label"] == 0]["capital_count"].mean()

# Create a bar chart
plt.figure(figsize=(6,4))
plt.bar(["Spam", "Ham"], [spam_avg, ham_avg], color=["red", "blue"])
plt.xlabel("Message Type")
plt.ylabel("Avg. Capitalized Words per Message")
plt.title("Capitalized Words in Spam vs Ham Messages")

print(f"Spam Avg: {spam_avg}")
print(f"Ham Avg: {ham_avg}")

# Check if values are non-zero
if spam_avg > 0 or ham_avg > 0:
    plt.bar(["Spam", "Ham"], [spam_avg, ham_avg], color=["red", "blue"])
    plt.xlabel("Message Type")
    plt.ylabel("Avg. Capitalized Words per Message")
    plt.title("Capitalized Words in Spam vs Ham Messages")
    plt.show()
else:
    print("No capitalized words detected.")

plt.show()

