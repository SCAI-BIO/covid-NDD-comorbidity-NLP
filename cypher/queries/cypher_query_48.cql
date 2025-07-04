#query disease ids from ols
import requests

# Define the base API endpoint
BASE_URL = "https://www.ebi.ac.uk/ols/api/search"

# Function to search for MONDO IDs for a given disease name
def search_mondo_ids(disease_name):
    params = {
        "q": disease_name,  # Query term (disease name)
        "ontology": "mondo",  # Specify the ontology
        "type": "class",  # Search for classes (terms)
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json().get("response", {}).get("docs", [])
        ids = []
        for result in results:
            mondo_id = result.get("obo_id")
            label = result.get("label")
            if mondo_id and label:
                ids.append({"id": mondo_id, "label": label})
        return ids
    else:
        print(f"Failed to fetch data for {disease_name}: {response.status_code}")
        return []

# List of diseases to search
diseases = [
    "Neurodegenerative disease",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Amyotrophic lateral sclerosis",
    "Multiple sclerosis",
    "Huntington's disease",
    "Lewy body dementia",
    "Frontotemporal dementia",
    "Creutzfeldt-Jakob disease",
    "Progressive supranuclear palsy",
    "COVID-19",
    "Post-COVID syndrome",
]

# Search for MONDO IDs for each disease
all_mondo_ids = []
for disease in diseases:
    print(f"Searching for MONDO IDs for: {disease}")
    ids = search_mondo_ids(disease)
    all_mondo_ids.extend(ids)

# Print the results
print("\nRetrieved MONDO IDs:")
for item in all_mondo_ids:
    print(f"{item['label']}: {item['id']}")

# Optional: Save the results to a file
import csv
with open("mondo_ids.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["label", "id"])
    writer.writeheader()
    writer.writerows(all_mondo_ids)
print("\nResults saved to 'mondo_ids.csv'")