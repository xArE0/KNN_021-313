import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load dataset
df = pd.read_csv("mixed_knn_data.csv")  # Use full path if needed

# Manhattan distance function for any number of features
def manhattan_distance(x1, x2):
    return sum(abs(a - b) for a, b in zip(x1, x2))

# Generalized KNN function
def knn_classify(new_item, k, dataset, feature_columns):
    distances = []
    for i, row in dataset.iterrows():
        features = [row[col] for col in feature_columns]
        dist = manhattan_distance(new_item, features)
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])
    top_k = distances[:k]
    categories = [dataset.iloc[i]['Category'] for i, _ in top_k]
    freq = Counter(categories)
    prediction = freq.most_common(1)[0][0]

    print(f"\nNew Item: {new_item}")
    print(f"Predicted Category: {prediction}")
    return prediction, distances, categories

# Test the KNN function with a new item
feature_columns = ['Size', 'TextureValue']
new_item = [2, 5]  # Example input (Size = 2, Texture = Medium)
k = 5

predicted_class, distance_list, top_categories = knn_classify(new_item, k, df, feature_columns)

# Visualization of Top 10 distances
plt.figure(figsize=(10, 5))
top_n = 10
top_indices = [i for i, _ in distance_list[:top_n]]
x_labels = [f"ID {df.iloc[i]['ID']}" for i in top_indices]
y_values = [dist for _, dist in distance_list[:top_n]]
top_k_categories = [df.iloc[i]['Category'] for i in top_indices]

colors = ['green' if i < k else 'orange' for i in range(top_n)]
bars = plt.bar(x_labels, y_values, color=colors)

# Category labels above bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
             top_k_categories[i], ha='center', fontsize=9)

plt.title("Top 10 Manhattan Distances to New Item")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.xticks(rotation=45)
plt.ylim(0, max(y_values) * 1.3)
plt.tight_layout()
plt.show()
