# Complete code for reading the covid-ndd PRIMEKG file, calculating statistics, and generating charts

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Function to clean and extract the first part of the node label
def clean_label(text):
    if isinstance(text, str) and '[' in text:
        return text.strip('[]"').split(",")[0].strip()
    return text

# Read the updated file with correct column names
file_path = 'data/primekg/final-primekg-covid-ndd.csv'
data = pd.read_csv(file_path)

# Clean and extract labels for Subject and Object
data['Subject_Label'] = data['Subj_label'].apply(clean_label)
data['Object_Label'] = data['Obj_label'].apply(clean_label)

# Calculate node label distribution
combined_labels = pd.concat([data['Subject_Label'], data['Object_Label']])
node_label_distribution = combined_labels.value_counts()

# Calculate relationship distribution
relation_distribution = data['Relation'].value_counts()

# Calculate degree centrality
node_degrees = Counter(data['Subject'].dropna()) + Counter(data['Object'].dropna())
top_10_high_degree_nodes = node_degrees.most_common(10)
top_10_high_degree_df = pd.DataFrame(top_10_high_degree_nodes, columns=["Node", "Degree"])

# Count number of unique nodes and triples
unique_nodes = len(set(data['Subject'].dropna()).union(set(data['Object'].dropna())))
total_triples = len(data)

# Calculate graph density
graph_density = 2 * total_triples / (unique_nodes * (unique_nodes - 1))

# Display statistics
print(f"Total unique nodes: {unique_nodes}")
print(f"Total number of triples: {total_triples}")
print(f"Graph density: {graph_density:.6f}")

# Plot charts
# 1. Node Label Distribution
plt.figure(figsize=(10, 6))
node_label_distribution.plot(kind='bar', color='lightblue', alpha=0.8)
plt.title("Node Types and Their Frequencies for PRIMEKG")
plt.xlabel("Node Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Relationship Distribution
plt.figure(figsize=(10, 6))
relation_distribution.plot(kind='bar', color='green', alpha=0.8)
plt.title("Relationship Types and Their Frequencies for PRIMEKG")
plt.xlabel("Relationship Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Top 10 High Degree Centrality Nodes
plt.figure(figsize=(10, 6))
top_10_high_degree_df.plot(kind='bar', x='Node', y='Degree', legend=False, color='skyblue', alpha=0.8)
plt.title("Top 10 Degree Centrality for PRIMEKG")
plt.xlabel("Node")
plt.ylabel("Degree Centrality")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()