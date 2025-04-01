import pandas as pd
import re
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Drop extra columns and rename
data = data.iloc[:, :2]
data.columns = ["label", "text"]

# Convert labels to binary (ham = 0, spam = 1)
data["label"] = data["label"].map({"ham": 0, "spam": 1})


def count_capitalized_words(text):
    return len(re.findall(r'\b[A-Z]{2,}\b', text))

# Apply function to dataset
data["capital_count"] = data["text"].apply(count_capitalized_words)

# Print sample messages with detected capitalized word counts
print(data[["text", "capital_count"]].head(10))

# Calculate average capitalized words per message for spam and ham
spam_avg = data[data["label"] == 1]["capital_count"].mean()
ham_avg = data[data["label"] == 0]["capital_count"].mean()

print(f"Spam Avg: {spam_avg}")
print(f"Ham Avg: {ham_avg}")

# Plot the results
plt.figure(figsize=(6,4))


plt.bar(["Spam", "Ham"], [spam_avg, ham_avg], color=["red", "blue"])
plt.xlabel("Message Type")
plt.ylabel("Avg. Capitalized Words per Message")
plt.title("Capitalized Words in Spam vs Ham Messages")
plt.show()

# NOTE K-means