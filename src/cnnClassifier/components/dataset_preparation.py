import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
import pandas as pd
from pathlib import Path
from collections import Counter
import os
from shutil import copy2
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

from cnnClassifier.entity.config_entity import PrepareDatasetConfig

class DataPreparation:
    def __init__(self, config: PrepareDatasetConfig):
        self.config = config

    def _get_class_names(self) -> List[str]:
        """Get sorted list of class names from directory structure."""
        return sorted([item.name for item in self.config.target_dir.glob('*') if item.is_dir()])


    def get_top_k_classes(self):
        self.df = pd.read_csv(self.config.csv_file_path, on_bad_lines='skip')
        self.df = self.df.dropna(subset=[self.config.label_col])  # Ensure no missing labels or image IDs
        top_classes = self.df[self.config.label_col].value_counts().nlargest(self.config.params_num_classes).index.tolist()
        return top_classes
    
    def organize_images_by_class(self,filtered_df):
        os.makedirs(self.config.target_dir, exist_ok=True)

        for _, row in filtered_df.iterrows():
            img_name = str(row['id']) + ".jpg"
            label = row[self.config.label_col]

            src = os.path.join(self.config.images_dir, img_name)
            dst_dir = os.path.join(self.config.target_dir, label)
            dst = os.path.join(dst_dir, img_name)

            if os.path.exists(src):  # Only copy if image exists
                os.makedirs(dst_dir, exist_ok=True)
                copy2(src, dst)
    
    def filter_df_by_classes(self,top_classes):
        return self.df[self.df[self.config.label_col].isin(top_classes)]
    

    def prepare_dataset(self):
        top_classes = self.get_top_k_classes()
        print("Top 10 Classes:", top_classes)

        TARGET_DIR = self.config.target_dir

        self.filtered_df = self.filter_df_by_classes(top_classes)
        self.organize_images_by_class(self.filtered_df)
        print(f"Organized images into: {TARGET_DIR}")
    
    
    @staticmethod
    def _preprocess_image(image_path: str, label: int, IMG_HEIGHT:int=150, IMG_WIDTH:int=150,no_of_classes:int = 10) -> Tuple[tf.Tensor, int]:
        """Load and preprocess a single image."""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        image = image / 255.0  # Normalize to [0, 1]
        #label = tf.one_hot(tf.cast(label, tf.int32), depth=no_of_classes)
        return image, label
    
    def _create_dataset(self) -> tf.data.Dataset:
        """Create and shuffle the full dataset."""
        class_names = self.get_top_k_classes()
        all_image_paths = []
        all_image_labels = []
        
        for label, class_name in enumerate(class_names):
            class_dir = self.config.target_dir / class_name
            image_paths = (list(class_dir.glob('*.jpg')) + 
                          list(class_dir.glob('*.jpeg')) + 
                          list(class_dir.glob('*.png')))
            image_paths = image_paths[:self.config.max_images]
            all_image_paths.extend([str(p) for p in image_paths])
            all_image_labels.extend([label] * len(image_paths))
            
        return tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels)) \
                             .shuffle(buffer_size=len(all_image_paths), seed=123)
    
    def _split_dataset(self, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Split dataset into train, validation and test sets."""
        dataset_size = len(dataset)
        train_size = int(self.config.trainset_size * dataset_size)
        val_size = int(self.config.valset_size * dataset_size)
        
        train_ds = dataset.take(train_size)
        remaining_ds = dataset.skip(train_size)
        val_ds = remaining_ds.take(val_size)
        test_ds = remaining_ds.skip(val_size)
        
        return train_ds, val_ds, test_ds
    
    def _configure_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply preprocessing, batching and prefetching to a dataset."""
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.params_batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # Convert labels to one-hot encoding
        dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=len(self.class_names))))
        return dataset
    
    def generate_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Generate and return train, validation and test datasets."""
        self.class_names = self._get_class_names()
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        full_ds = self._create_dataset()
        train_ds, val_ds, test_ds = self._split_dataset(full_ds)
        
        self.train_ds = self._configure_dataset(train_ds)
        self.val_ds = self._configure_dataset(val_ds)
        self.test_ds = self._configure_dataset(test_ds)
        
        self._print_dataset_stats()
        return self.train_ds, self.val_ds, self.test_ds


    def _print_dataset_stats(self):
        """Print statistics about the generated datasets."""
        print("\nDataset sizes:")
        print(f"Train: {tf.data.experimental.cardinality(self.train_ds).numpy() * self.config.params_batch_size} images (approx)")
        print(f"Validation: {tf.data.experimental.cardinality(self.val_ds).numpy() * self.config.params_batch_size} images (approx)")
        print(f"Test: {tf.data.experimental.cardinality(self.test_ds).numpy() * self.config.params_batch_size} images (approx)")
        print("Note: Cardinality is reported in batches for prefetched datasets.")
    
