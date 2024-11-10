import os
import queue
import numpy as np

class Trainer:
    def __init__(self, model: Model, config: dict, test=False) -> None:

        train_config = config["train"]
        model_dir = train_config["model_dir"]
        # If model continue, continues model from the latest checkpoint
        model_continue = train_config.get("continue", True)
        # If the directory has not been created, can't continue from anything
        if not os.path.isdir(model_dir):
            model_continue = False
        if test:
            model_continue = True

