import json

# Function to load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to validate the mapping between the two files
def validate_flower_mapping(flatten_file, mapping_file):
    flatten_data = load_json(flatten_file)
    mapping_data = load_json(mapping_file)

    # Flatten the mapping file to get a set of all classified flowers
    classified_items = set()
    for category, flowers in mapping_data.items():
        classified_items.update(flowers)

    # Extract all flower names from the flatten file
    flatten_values = set(flatten_data.values())

    # Check if all flowers are classified
    missing = flatten_values - classified_items
    if missing:
        print(f"Error: The following flowers are missing from the classification: {missing}")
    
    # Check if there are extra flowers in the classification
    extra = classified_items - flatten_values
    if extra:
        print(f"Error: The following flowers are extra in the classification: {extra}")
    
    # Check for unique classification (i.e., no duplicates in different categories)
    flower_to_category = {}
    for category, flowers in mapping_data.items():
        for flower in flowers:
            if flower in flower_to_category:
                print(f"Error: '{flower}' appears in multiple categories: '{flower_to_category[flower]}' and '{category}'")
            else:
                flower_to_category[flower] = category

    # Print validation result
    if not missing and not extra and len(flower_to_category) == len(classified_items):
        print("The mapping is valid.")
    else:
        print("The mapping is invalid.")

# Paths to files
flatten_file = 'classes/flower_102_mapping_flatten.json'
mapping_file = 'classes/flower_102_mapping.json'

# Validate the mapping
validate_flower_mapping(flatten_file, mapping_file)
