import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit

from aucmedi.data_processing.io_data import input_interface
from aucmedi.data_processing import augmentation, data_generator
from aucmedi.data_processing.io_loader.sitk_loader import sitk_loader



# ====== # 
# CONFIG # 
# ====== # 

CONFIG = {
    "img_dir": "/path/to/your/img/folder",
    "csv_file": "yourfile.csv" ,
    "batch_size": 32,
    "workers": 32,
    "resize": (193, 229, 193),
    "seed": 123,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
}


# ============ #
# DATA LOADING #
# ============ #

def load_dataset(img_dir, csv_file):
    df = pd.read_csv(csv_file)

    print("\nClass distribution (raw):")
    print(df["Original Class"].value_counts())

    loader = input_interface(
        interface="csv",
        path_imagedir=img_dir,
        path_data=csv_file,
        col_sample="Filename",
        col_class="Original Class",
        training=True,
        ohe=False,
    )

    images, labels, n_classes, class_names, image_format = loader

    print("\nDataset loaded")
    print(f"Images: {len(images)}")
    print(f"Label shape: {labels.shape}")
    print(f"Classes: {class_names}")

    return images, labels, class_names, image_format


# ================== #
# CLASS DISTRIBUTION #
# ================== #

def print_class_distribution(labels, class_names):
    stats = []

    for i in range(labels.shape[1]):
        count = np.sum(labels[:, i])
        name = class_names[i] if class_names else str(i)
        perc = round(count / labels.shape[0] * 100, 2)
        stats.append([name, int(count), perc])

    df_stats = pd.DataFrame(stats, columns=["class", "count", "percentage"])
    print("\nClass percentage:")
    print(df_stats)

    return df_stats


# ===================== #
# GROUPING (NO LEAKAGE) #
# ===================== #

def extract_group_id(filename: str) -> str:

    return filename[4:] if filename.startswith("aug_") else filename


def create_groups(image_list):
    return np.array([extract_group_id(f) for f in image_list])


def leakage_safe_split(images, labels, groups, seed=123):
    """
    70/15/15 split using group-aware splitting.
    """

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )

    train_idx, temp_idx = next(
        gss1.split(images, labels, groups)
    )

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=seed
    )

    temp_images = np.array(images)[temp_idx]
    temp_labels = labels[temp_idx]
    temp_groups = groups[temp_idx]

    val_idx_rel, test_idx_rel = next(
        gss2.split(temp_images, temp_labels, temp_groups)
    )

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    return train_idx, val_idx, test_idx


# ====================== #
# DATA GENERATOR BUILDER #
# ====================== #

def build_generator(samples, labels, img_dir, cfg, augment=False):

    return data_generator.DataGenerator(
        samples=samples,
        labels=labels,
        path_imagedir=img_dir,
        image_format=None,
        loader=sitk_loader,
        shuffle=augment,
        batch_size=cfg["batch_size"],
        workers=cfg["workers"],
        grayscale=True,
        data_aug=augmentation.VolumeAugmentation(
            flip=True,
            rotate=True,
            brightness=False,
            contrast=False,
            scale=True
        ) if augment else None,
        resize=cfg["resize"],
        standardize_mode="minmax",
        seed=cfg["seed"],
    )


# ============= #
# MAIN PIPELINE #
# ============= #

def main(cfg):

    images, labels, class_names, _ = load_dataset(
        cfg["img_dir"],
        cfg["csv_file"]
    )

    print_class_distribution(labels, class_names)
    groups = create_groups(images)

    train_idx, val_idx, test_idx = leakage_safe_split(
        images, labels, groups, cfg["seed"]
    )

    train_images, train_labels = np.array(images)[train_idx], labels[train_idx]
    val_images, val_labels = np.array(images)[val_idx], labels[val_idx]
    test_images, test_labels = np.array(images)[test_idx], labels[test_idx]

    # Generators
    cnn_train_gen = build_generator(train_images, train_labels, cfg["img_dir"], cfg, augment=True)
    cnn_val_gen   = build_generator(val_images, val_labels, cfg["img_dir"], cfg, augment=False)
    cnn_test_gen  = build_generator(test_images, test_labels, cfg["img_dir"], cfg, augment=False)

    int_labels = np.argmax(train_labels, axis=1)
    named = [class_names[i] for i in int_labels]

    print("\nTrain label distribution:")
    print(Counter(named))

    return cnn_train_gen, cnn_val_gen, cnn_test_gen, labels, class_names


# === #
# RUN #
# === #

if __name__ == "__main__":
    main(CONFIG)
