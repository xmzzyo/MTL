# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_prepare
   Description :
   Author :       xmz
   date：          19-4-16
-------------------------------------------------
"""
import os
import random
import re

TASKS_NAME = ["apparel",
              "baby",
              "books",
              "camera_photo",
              "electronics",
              "health_personal_care",
              "imdb",
              "kitchen_housewares",
              "magazines",
              "music",
              "software",
              "sports_outdoors",
              "toys_games",
              "video"
              ]


def split_sentiment_review():
    print(os.path.dirname(__file__))
    dir_path = "../../data/mtl-dataset"
    print(os.listdir(dir_path))
    valid_proportion = 0.1
    for task in TASKS_NAME:
        filename = os.path.join(dir_path, task + ".task.train")
        train_set = []
        with open(filename, "r") as f:
            for line in f.readlines():
                if line is None:
                    continue
                train_set.append(line.strip())
        # os.rename(filename, filename + ".bak")
        val_size = int(len(train_set) * valid_proportion)
        print(val_size)
        random.shuffle(train_set)
        val_set = train_set[-val_size:]
        train_set = train_set[:-val_size]
        with open(filename, "w") as f:
            f.write("\n".join(train_set))
        with open(re.sub("train", "val", filename), "w") as f:
            f.write("\n".join(val_set))
        print(task + " splited")


if __name__ == "__main__":
    split_sentiment_review()
