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

from cnnClassifier.entity.config_entity import TrainingConfig
                                                
import tensorflow.python.keras.models as Model

class Training:
    def __init__(self, config: TrainingConfig,train_ds,valid_ds):
        self.config = config
        self.train_generator = train_ds
        self.valid_generator = valid_ds
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def save_model(path: Path, model: Model):
        tf.keras.models.save_model(model, path)
        model.save(path)

    def train(self):
        self.get_base_model()

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            validation_data=self.valid_generator,
            validation_steps=self.config.params_batch_size,
            
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )