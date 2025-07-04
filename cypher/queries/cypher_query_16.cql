import pandas as pd
import os
import re
from fuzzywuzzy import fuzz  # Alternatively, you can use rapidfuzz

# List of common suffixes/terms to remove
suffixes_to_remove = ["disease", "disorder", "syndrome", "infection", "virus", "condition", r"\d+"]  # Includes numbers (e.g., "19")

# Function to normalize nodes (lowercase, remove punctuation, strip leading/trailing spaces, remove suffixes but preserve spaces)
def normalize_node(node):
    # Convert to lowercase
    node = node.lower()
    
    # Remove possessive apostrophes
    node = re.sub(r"'s", "", node)
    
    # Remove punctuation except spaces
    node = re.sub(r'[^\w\s]', '', node)
    
    # Remove common suffixes
    for suffix in suffixes_to_remove:
        node = re.sub(rf"\b{suffix}\b", "", node)  # Remove the suffix
    
    # Remove extra spaces from leading/trailing
    node = node.strip()
    
    return node

# Function to perform fuzzy matching and identify similar nodes
def fuzzy_unique_nodes(node_list, threshold=80):
    unique_nodes = []
    
    for node in node_list:
        matched = False
        for unique_node in unique_nodes:
            if fuzz.ratio(node, unique_node) >= threshold:
                matched = True
                break
        if not matched:
            unique_nodes.append(node)
    
    return unique_nodes

# Load all the provided CSV files
file_paths = [
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/opentargets+disgenet+drugbank+indra-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/cbm-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/pubtator-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/scaiDMaps-comorbidity-paths-3-hops.csv',
    r'C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results/sherpa-comorbidity-paths-3-hops.csv',
    r"C:\Users\nbabaiha\Documents\GitHub\COMMUTE\commute\neo4j-import-analysis\neo4j-analysis-results\primekg-comorbidity-paths-3-hops.csv"
]

# Load the dataframes into a dictionary for processing
dfs = {file_path: pd.read_csv(file_path) for file_path in file_paths}

# Function to normalize and extract nodes and paths
def normalize_and_extract_nodes_paths(df, node_col='Nodes'):
    all_nodes = []
    all_paths = []
    
    # Extract individual nodes and paths from the series of nodes in every path
    for path in df[node_col]:
        nodes = eval(path)  # Convert string to list
        normalized_nodes = [normalize_node(node) for node in nodes]  # Normalize each node
        all_nodes.extend(normalized_nodes)  # Add normalized nodes to the list
        normalized_path = tuple(normalized_nodes)  # Store path as a tuple (immutable) for uniqueness
        all_paths.append(normalized_path)
    
    # Apply fuzzy matching to the nodes list to group similar nodes
    unique_fuzzy_nodes = fuzzy_unique_nodes(all_nodes)
    
    return unique_fuzzy_nodes, all_paths

# Step 1: Collect all normalized and fuzzy-matched nodes and paths from each source
all_nodes_paths = {}
for file_path in dfs:
    nodes, paths = normalize_and_extract_nodes_paths(dfs[file_path])
    all_nodes_paths[file_path] = {
        "Nodes": nodes,
        "Paths": paths,
        "Total Paths": len(paths)  # Count total number of paths for this source
    }

# Step 2: Identify unique nodes and paths for each source (compared to all other sources)
unique_summary = {}
all_sources_nodes = {file_path: set(all_nodes_paths[file_path]["Nodes"]) for file_path in all_nodes_paths}
all_sources_paths = {file_path: set(all_nodes_paths[file_path]["Paths"]) for file_path in all_nodes_paths}

for file_path in dfs:
    # Unique nodes for this source: Nodes in this source but not in any other source
    other_sources_nodes = set().union(*[all_sources_nodes[other] for other in all_sources_nodes if other != file_path])
    unique_nodes = all_sources_nodes[file_path] - other_sources_nodes

    # Unique paths for this source: Paths in this source but not in any other source
    other_sources_paths = set().union(*[all_sources_paths[other] for other in all_sources_paths if other != file_path])
    unique_paths = all_sources_paths[file_path] - other_sources_paths

    unique_summary[file_path] = {
        "Unique Nodes": list(unique_nodes),
        "Unique Nodes Count": len(unique_nodes),
        "Unique Paths": list(unique_paths),
        "Unique Paths Count": len(unique_paths),
        "Total Paths": all_nodes_paths[file_path]["Total Paths"]  # Total number of paths
    }

# Step 3: Identify common nodes and paths (present in all sources)
common_nodes = set.intersection(*all_sources_nodes.values())
common_paths = set.intersection(*all_sources_paths.values())

# Create a summary DataFrame for unique node, path counts, and total paths
summary_data = {
    file_path: [
        unique_summary[file_path]["Unique Nodes Count"],
        unique_summary[file_path]["Unique Paths Count"],
        unique_summary[file_path]["Total Paths"]
    ]
    for file_path in unique_summary
}
summary_df = pd.DataFrame(summary_data, index=["Unique Nodes Count", "Unique Paths Count", "Total Paths"]).T

# Create a combined DataFrame for all unique nodes from each source
unique_nodes_combined = pd.DataFrame()
for file_path in unique_summary:
    base_name = os.path.basename(file_path).split('.')[0][:20]  # Limit to 20 characters
    unique_nodes_combined[base_name] = pd.Series(unique_summary[file_path]["Unique Nodes"])

# Save the summary, unique nodes, unique paths, and common elements to an Excel file
output_file_path_summary = 'all_paths_3hops_summary.xlsx'
with pd.ExcelWriter(output_file_path_summary) as writer:
    # Save the summary data
    summary_df.to_excel(writer, sheet_name="Summary")
    
    # Save unique nodes and paths for each source into separate sheets
    for file_path in unique_summary:
        base_name = os.path.basename(file_path).split('.')[0][:20]  # Limit to 20 characters
        
        # Save unique nodes
        nodes_df = pd.DataFrame(unique_summary[file_path]["Unique Nodes"], columns=["Unique Nodes"])
        nodes_df.to_excel(writer, sheet_name=base_name + "_un_nodes")
        
        # Save unique paths
        paths_df = pd.DataFrame([' -> '.join(path) for path in unique_summary[file_path]["Unique Paths"]], columns=["Unique Paths"])
        paths_df.to_excel(writer, sheet_name=base_name + "_un_paths")
    
    # Save the common nodes across all sources
    common_nodes_df = pd.DataFrame(list(common_nodes), columns=["Common Nodes"])
    common_nodes_df.to_excel(writer, sheet_name="Common Nodes")
    
    # Save the common paths across all sources
    common_paths_df = pd.DataFrame([' -> '.join(path) for path in common_paths], columns=["Common Paths"])
    common_paths_df.to_excel(writer, sheet_name="Common Paths")
    
    # Save the combined unique nodes from all sources in one sheet
    unique_nodes_combined.to_excel(writer, sheet_name="All Unique Nodes")

# Print the summary to the console
print(summary_df)

# Output the file path for the saved Excel file
print(f"Summary, unique nodes, paths, and common elements saved to: {output_file_path_summary}")