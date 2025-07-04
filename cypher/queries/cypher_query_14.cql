# Use path files extracted from Neo4js, this Function to extract unique nodes from the 'Nodes' column
import pandas as pd

# Load the two CSV files into pandas DataFrames
file_path_all = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/all-dbs-comorbidity-paths-3-hops.csv"
file_path_sherpa = r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-db-analysis-textmining-databases/sherpa-comorbidity-paths-3-hops.csv"

df_all = pd.read_csv(file_path_all)
df_sherpa = pd.read_csv(file_path_sherpa)

# Function to extract unique nodes from the 'Nodes' column
def extract_unique_nodes(df):
    # Flatten the list of nodes and extract unique values
    nodes_series = df['Nodes'].apply(lambda x: eval(x))  # Convert string representation of list to list
    unique_nodes = set([node for sublist in nodes_series for node in sublist])
    return unique_nodes

# Extract unique nodes from both datasets
unique_nodes_all = extract_unique_nodes(df_all)
unique_nodes_sherpa = extract_unique_nodes(df_sherpa)

# Convert nodes to lowercase for case-insensitive comparison
unique_nodes_all_lower = {node.lower() for node in unique_nodes_all}
unique_nodes_sherpa_lower = {node.lower() for node in unique_nodes_sherpa}

# Function to perform fuzzy matching with a similarity threshold
def fuzzy_match_nodes(set1, set2, threshold=85):
    matched_nodes = set()
    unmatched_set1 = set()
    unmatched_set2 = set(set2)  # Track which nodes in set2 are unmatched

    for node1 in set1:
        # Find the best match from set2 for node1 using fuzzy matching
        best_match, score = process.extractOne(node1, set2, scorer=fuzz.ratio)
        
        # If the score exceeds the threshold, consider it a match
        if score >= threshold:
            matched_nodes.add((node1, best_match, score))
            unmatched_set2.discard(best_match)  # Remove matched node from unmatched set2
        else:
            unmatched_set1.add(node1)

    return matched_nodes, unmatched_set1, unmatched_set2

# Perform fuzzy matching
fuzzy_matched_nodes, fuzzy_unmatched_all, fuzzy_unmatched_sherpa = fuzzy_match_nodes(unique_nodes_all_lower, unique_nodes_sherpa_lower)

# Function to extract paths as a tuple of (Nodes, Relationships)
def extract_paths(df):
    paths_series = df.apply(lambda row: (tuple(eval(row['Nodes'])), tuple(eval(row['Relationships']))), axis=1)
    return set(paths_series)

# Extract paths from both datasets
paths_all = extract_paths(df_all)
paths_sherpa = extract_paths(df_sherpa)

# Find common paths and unique paths in both datasets
common_paths = paths_all.intersection(paths_sherpa)
unique_paths_all_only = paths_all - paths_sherpa
unique_paths_sherpa_only = paths_sherpa - paths_all

# Updated print function to print nodes in a structured format
def print_nodes(title, node_list):
    print(f"\n{title} ({len(node_list)}):")
    for idx, node in enumerate(sorted(node_list), 1):
        print(f"  {idx}. {node}")

# Display structured results
print(f"Fuzzy Matched Nodes ({len(fuzzy_matched_nodes)}):")
for idx, match in enumerate(fuzzy_matched_nodes, 1):
    print(f"  {idx}. {match[0]} (from 'all-dbs-comorbidity-paths-3-hops.csv') matches {match[1]} (from 'sherpa-comorbidity-paths-3-hops.csv') with {match[2]}% similarity")

# Print unique nodes from both datasets
print_nodes("Unique Nodes in 'all-dbs-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_all)
print_nodes("Unique Nodes in 'sherpa-comorbidity-paths-3-hops.csv' (Unmatched)", fuzzy_unmatched_sherpa)

# Display the path results
print(f"\nCommon Paths: {len(common_paths)}")
print(f"Unique Paths in 'all-dbs-comorbidity-paths-3-hops.csv': {len(unique_paths_all_only)}")
print(f"Unique Paths in 'sherpa-comorbidity-paths-3-hops.csv': {len(unique_paths_sherpa_only)}")