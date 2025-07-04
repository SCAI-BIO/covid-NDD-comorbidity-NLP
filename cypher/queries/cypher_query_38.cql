#Get the paths that are unique for one json file (give a name) comapred to all paths
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
    all_results = {}
    for filename in glob.glob(file_pattern):
        print(f"Loading data from {filename}...")
        data = load_results_from_file(filename)
        if data:
            all_results[filename] = data  # Store each file's data with its filename as key
    return all_results

# Function to find paths unique to a specific JSON file
def find_unique_paths(target_filename, paths_dict):
    if target_filename not in paths_dict:
        print(f"File {target_filename} not found in loaded data.")
        return []
    
    # Convert paths to sets of tuples to ensure unique paths and to compare them
    target_file_paths = set(tuple(map(tuple, path)) for path in paths_dict[target_filename])
    
    # Combine paths from all other files
    other_files_paths = set()
    for filename, paths in paths_dict.items():
        if filename != target_filename:
            other_files_paths.update(tuple(map(tuple, path)) for path in paths)
    
    # Find paths unique to the target file by subtracting the other paths
    unique_paths = target_file_paths - other_files_paths

    # Convert the set of tuples back to lists
    unique_paths = [list(map(list, path)) for path in unique_paths]

    return unique_paths

# Load data from multiple JSON files
file_pattern = 'neo4j_results_*.json'  # Adjust the pattern to match your files
all_neo4j_results = load_all_json_files(file_pattern)

if not all_neo4j_results:
    print("No results loaded from files.")
else:
    # Specify the filename for which you want to find unique paths
    target_filename = 'neo4j_results_2.json'  # Replace with the specific filename
    
    # Use the find_unique_paths function on the loaded results
    unique_paths = find_unique_paths(target_filename, all_neo4j_results)
    
    print(f"Unique paths in the file {target_filename}: {unique_paths}")