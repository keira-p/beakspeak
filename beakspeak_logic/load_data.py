from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data/raw_data/CUB_200_2011")


def load_metadata(metadata_path):
    """
    Create single metadata df from raw data files.

    Args:
        metadata_path (Path): Path to the root of the CUB dataset.

    Returns:
        pd.DataFrame: Combined metadata dataframe.
    """

    # Load raw metadata files
    images = pd.read_csv(DATA_DIR / "images.txt", sep=" ", header=None, names=["image_id", "image_path"])
    labels = pd.read_csv(DATA_DIR / "image_class_labels.txt", sep=" ", header=None, names=["image_id", "class_id"])
    class_names = pd.read_csv(DATA_DIR / "classes.txt", sep=" ", header=None, names=["class_id", "class_name"])
    split = pd.read_csv(DATA_DIR / "train_test_split.txt", sep=" ", header=None, names=["image_id", "is_training_image"])

    # Merge into single dataframe
    metadata = (
        images
        .merge(labels, on="image_id")
        .merge(class_names, on="class_id")
        .merge(split, on="image_id")
    )

    # Add full file paths
    metadata["file_path"] = metadata["image_path"].apply(lambda x: DATA_DIR / "images" / x)

    # Translate binary train/test column to string labels
    metadata["split"] = metadata["is_training_image"].apply(lambda x: "train" if x == 1 else "test")

    return metadata


def split_data(metadata, test_size=0.2, random_state=42):
    """
    Split training data into train/val sets.

    Args:
        metadata (pd.DataFrame): Full metadata dataframe, including train/test split.
        test_size (float): Proportion of training data to use for validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_df, val_df, test_df (pd.DataFrame): DataFrames for training, validation, and test sets.
    """

    # Separate out test set
    test_df = metadata[metadata["split"] == "test"].copy().reset_index(drop=True)
    train_val_df = metadata[metadata["split"] == "train"].copy().reset_index(drop=True)

    # Split training data into train/val, maintaining class distribution
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_val_df["class_id"]
        )

    return train_df, val_df, test_df
