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

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity

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


def get_domain_embeddings_as_numpy():
    serialization_dir = "../data/out_md_lstm_0.005adv_de/"

    model_path = os.path.join(serialization_dir, "model_state.th")

    # model_state = torch.load(model_path, map_location=device_mapping(-1))
    # model.load_state_dict(model_state)

    state_dict = torch.load(model_path)

    for key, value in state_dict.items():
        print(key, value.shape)
        if "_domain_embeddings.weight" == key:
            de = value.cpu().numpy()
    print(de.shape)

    return de


domain_embedding = get_domain_embeddings_as_numpy()


def cal_sim():
    for i in range(len(tasks)):
        for j in range(i, len(tasks)):
            print(tasks[i], tasks[j], cosine_similarity([domain_embedding[i], domain_embedding[j]]))
            print(tasks[i], tasks[j], euclidean_distances([domain_embedding[i], domain_embedding[j]]))


def display():
    pca = PCA(n_components=2).fit_transform(domain_embedding)

    tsne = TSNE(n_components=2, learning_rate=20).fit_transform(domain_embedding)
    print(tsne.shape)
    plt.figure(figsize=(20, 12))
    plt.subplot(121)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=range(len(tasks)))
    for i in range(tsne.shape[0]):
        plt.text(tsne[i, 0], tsne[i, 1], tasks[i])
    plt.subplot(122)
    plt.scatter(pca[:, 0], pca[:, 1], c=range(len(tasks)))
    for i in range(pca.shape[0]):
        plt.text(pca[i, 0], pca[i, 1], tasks[i])
    # plt.colorbar()
    plt.savefig("tsne.png")
    plt.show()


def get_rank():
    print(np.linalg.matrix_rank(domain_embedding))


def add_noise():
    index = 1
    u, s, vh = np.linalg.svd(domain_embedding)
    smat = np.zeros(domain_embedding.shape, dtype=complex)
    smat[:14, :14] = np.diag(s)
    reconstruct = np.dot(u, np.dot(smat, vh))
    print(reconstruct.shape)
    print(np.allclose(domain_embedding, reconstruct))
    noise = 0.01 * np.random.normal(0.5, 1, 14)
    noise[index] = 0
    noise += s
    smat[:14, :14] = np.diag(noise)
    reconstruct = np.real(np.dot(u, np.dot(smat, vh)))
    print(reconstruct.shape)
    print(np.allclose(domain_embedding, reconstruct))
    print(np.mean(domain_embedding[index]), np.std(domain_embedding[index]))
    print(np.mean(np.subtract(domain_embedding[index], reconstruct[index])))

    de = np.append(domain_embedding, [reconstruct[index]], axis=0)
    print(de.shape)

    tsne = TSNE(n_components=2, learning_rate=20, random_state=11).fit_transform(de)
    pca = PCA(n_components=2).fit_transform(de)
    print(pca.shape)
    plt.figure(figsize=(20, 12))
    plt.subplot(121)
    tasks.append("noise")
    plt.scatter(pca[:, 0], pca[:, 1], c=range(len(tasks)))
    for i in range(pca.shape[0]):
        plt.text(pca[i, 0], pca[i, 1], tasks[i])
    # plt.subplot(122)
    # plt.scatter(tsne_noise[:, 0], tsne_noise[:, 1], c=range(len(tasks)))
    # for i in range(tsne_noise.shape[0]):
    #     plt.text(tsne_noise[i, 0], tsne_noise[i, 1], tasks[i])
    # plt.colorbar()
    plt.savefig("tsne.png")
    plt.show()


if __name__ == "__main__":
    add_noise()
