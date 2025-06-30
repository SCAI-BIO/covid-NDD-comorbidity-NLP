# Read all json paths and compare them all as a common path
import json
import glob

# Function to load data from a JSON file
def load_results_from_file(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filename}")
        return []

# Function to load and aggregate data from multiple JSON files
def load_all_json_files(file_pattern):
    all_results = []
    for filename in glob.glob(file_pattern):
        print(f"Loading data from {filename}...")
        data = load_results_from_file(filename)
        if data:
            all_results.append(data)  # Do not flatten, keep each file's data separate
    return all_results

# Function to compare paths between different graphs
def compare_paths(paths_list):
    if not paths_list:
        print("No paths to compare.")
        return set()
    
    # Convert paths to sets of tuples to ensure unique paths and to compare them
    path_sets = [set(tuple(map(tuple, path)) for path in paths) for paths in paths_list]
    
    # Find the intersection of all path sets
    common_paths = set.intersection(*path_sets)

    # Convert the set of tuples back to lists
    common_paths = [list(map(list, path)) for path in common_paths]

    return common_paths

# Load data from multiple JSON files
file_pattern = 'neo4j_results_*.json'  # Adjust the pattern to match your files
all_neo4j_results = load_all_json_files(file_pattern)

if not all_neo4j_results:
    print("No results loaded from files.")
else:
    # Use the compare_paths function on the loaded results
    common_paths = compare_paths(all_neo4j_results)
    
    print(f"Common paths across all graphs: {len(common_paths)}")