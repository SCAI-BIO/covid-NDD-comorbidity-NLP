import os

def print_filtered_tree(startpath, prefix=""):
    entries = sorted(os.listdir(startpath))
    # Filter out hidden folders and files except .py/.ipynb files
    filtered = []
    for e in entries:
        if e.startswith('.'):
            continue  # skip hidden files and folders like .git
        full_path = os.path.join(startpath, e)
        if os.path.isdir(full_path):
            filtered.append(e)
        elif e.endswith('.py') or e.endswith('.ipynb'):
            filtered.append(e)

    for i, entry in enumerate(filtered):
        path = os.path.join(startpath, entry)
        connector = "├── " if i < len(filtered) - 1 else "└── "
        print(prefix + connector + entry)
        if os.path.isdir(path):
            extension = "│   " if i < len(filtered) - 1 else "    "
            print_filtered_tree(path, prefix + extension)

root_folder = os.path.basename(os.getcwd())
print(root_folder)
print_filtered_tree('.')
