# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     domain_embedding
   Description :
   Author :       xmz
   date：          2019/4/21
-------------------------------------------------
"""
import os

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

serialization_dir = "../data/out_md_cnn_14_f_de_50/"

model_path = os.path.join(serialization_dir, "model_state.th")

# model_state = torch.load(model_path, map_location=device_mapping(-1))
# model.load_state_dict(model_state)

tasks = ["apparel",
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

state_dict = torch.load(model_path)

for key, value in state_dict.items():
    if "_domain_embeddings.weight" == key:
        domain_embedding = value.cpu().numpy()

pca = PCA(n_components=2).fit_transform(domain_embedding)

tsne = TSNE(n_components=2, learning_rate=20).fit_transform(domain_embedding)
print(tsne.shape)
plt.figure(figsize=(20, 12))
plt.subplot(121)
plt.scatter(tsne[:, 0], tsne[:, 1], c=range(len(tasks)))
for i in range(tsne.shape[0]):
    plt.text(tsne[i, 0], tsne[i, 1], tasks[i])
# plt.text(tsne[:, 0], tsne[:, 1], TASKS_NAME)
plt.subplot(122)
plt.scatter(pca[:, 0], pca[:, 1], c=range(len(tasks)))
for i in range(pca.shape[0]):
    plt.text(pca[i, 0], pca[i, 1], tasks[i])
# plt.text(pca[:, 0], pca[:, 1], TASKS_NAME)
# plt.colorbar()
plt.savefig("tsne_cnn.png")
plt.show()

exit()
