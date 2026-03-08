import os
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPECIES_RAW = os.path.join(BASE_DIR, "species_raw")
SPECIES_CLEAN = os.path.join(BASE_DIR, "species_clean")

DISEASE_RAW = os.path.join(BASE_DIR, "disease_raw")
DISEASE_CLEAN = os.path.join(BASE_DIR, "disease_clean")

REMOVE_CLASS = "Type 4 Indian Carp"


# -----------------------------
# REMOVE CORRUPTED IMAGES
# -----------------------------

def remove_corrupted_images(dataset_path):

    print(f"\nChecking corrupted images in {dataset_path}")

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            img_path = os.path.join(root, file)

            try:
                Image.open(img_path).verify()

            except:
                print("Removing corrupted:", img_path)
                os.remove(img_path)


# -----------------------------
# SPLIT DATASET
# -----------------------------

def split_dataset(raw_path, output_path):

    splits = ["train", "val", "test"]

    for split in splits:
        os.makedirs(os.path.join(output_path, split), exist_ok=True)

    for class_name in os.listdir(raw_path):

        class_path = os.path.join(raw_path, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            file_name for file_name in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, file_name))
        ]

        if len(images) == 0:
            print(f"Skipping empty class: {class_name}")
            continue

        if len(images) < 3:
            print(f"Skipping class with too few images: {class_name} ({len(images)})")
            continue

        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        for split, split_imgs in zip(["train","val","test"],[train_imgs,val_imgs,test_imgs]):

            split_class_path = os.path.join(output_path, split, class_name)

            os.makedirs(split_class_path, exist_ok=True)

            for img in tqdm(split_imgs, desc=f"{class_name} -> {split}"):

                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_path, img)

                shutil.copy(src, dst)


def main():
    # -----------------------------
    # SPECIES DATASET PREP
    # -----------------------------

    print("\nPreparing Species Dataset")

    # remove indian carp
    carp_path = os.path.join(SPECIES_RAW, REMOVE_CLASS)

    if os.path.exists(carp_path):
        shutil.rmtree(carp_path)
        print("Removed class:", REMOVE_CLASS)

    remove_corrupted_images(SPECIES_RAW)

    split_dataset(SPECIES_RAW, SPECIES_CLEAN)

    # -----------------------------
    # DISEASE DATASET PREP
    # -----------------------------

    print("\nPreparing Disease Dataset")

    remove_corrupted_images(DISEASE_RAW)

    split_dataset(DISEASE_RAW, DISEASE_CLEAN)

    print("\nDatasets prepared successfully!")


if __name__ == "__main__":
    main()
