from pathlib import Path
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data/raw_data/CUB_200_2011")

from beakspeak.params import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, AUTOTUNE, SEED


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


def load_and_preprocess_image(file_path, label):
    """
    Load and preprocess a single image.

    Inputs:
    - file_path: Path to the image file
    - label: Integer class label

    Outputs:
    - image: Preprocessed image tensor of shape (224, 224, 3)
    - label: Unchanged integer class label
    """

    # Read raw file
    image = tf.io.read_file(file_path)

    # Decode JPEG into 3-channel RGB image
    image = tf.image.decode_jpeg(image, channels=3)

    # Set image shape
    image.set_shape([None, None, 3])

    # Resize for CNN processing and preserve aspect ratio
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], preserve_aspect_ratio=True)

    # Pad to ensure final size is (224, 224)
    image = tf.image.resize_with_pad(image, IMG_HEIGHT, IMG_WIDTH)

    # Scale pixel values to [0, 1]
    image = image / 255.0

    return image, label


def create_dataset(df, shuffle=True):
    """
    Create a TensorFlow dataset from a DataFrame of image paths and labels.

    Parameters:
    - df: DataFrame with 'file_path' and 'label' columns
    - shuffle: True for train, False for val/test

    Returns:
    - A batched tf.data.Dataset object
    """

    # Create dataset from pairs of image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            df["file_path"].astype(str).values,
            df["label"].values
        )
    )

    # Shuffle for training across length of training set
    # Not for validation/test
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED
        )

    # Map the loading and preprocessing function across the dataset
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE
    )

    # Batch the dataset for efficient training
    dataset = dataset.batch(BATCH_SIZE)

    # Prefetch to improve performance
    # Overlapping data preprocessing and model execution
    # (Fetches the next batch while the current batch is being processed)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset
