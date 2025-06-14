import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config



    @staticmethod
    def load_model(path: Path):
        return tf.keras.models.load_model(path)
    

    def evaluation(self,test_ds):
        print("Loading model from path:", self.config.path_of_model)
        self.model = self.load_model(self.config.path_of_model)
        self.score = self.model.evaluate(test_ds)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
