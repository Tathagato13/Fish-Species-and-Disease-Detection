import os

DATASET_PATH = "datasets/species_clean"

for split in ["train", "val", "test"]:
    
    print("\n", split.upper())
    
    split_path = os.path.join(DATASET_PATH, split)
    
    for cls in os.listdir(split_path):
        
        class_path = os.path.join(split_path, cls)
        
        count = len(os.listdir(class_path))
        
        print(f"{cls}: {count}")