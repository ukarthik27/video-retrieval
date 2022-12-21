import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.manifold import TSNE

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(trainloader, image_head, text_head, image_optimizer, text_optimizer, criterion, only_image=False, only_text=False):
    image_head.train()
    text_head.train()
    train_loss = 0
    num_samples = 0

    for batch in trainloader:
        if not only_text:
            image_optimizer.zero_grad()
        if not only_image:
            text_optimizer.zero_grad()
        image_embedding = image_head(batch["images"].to(device))
        text_embedding = text_head(batch["texts"].to(device))
        loss = criterion(image_embedding, text_embedding, batch["labels"].to(device))
        loss.backward()
        if not only_text:
            image_optimizer.step()
        if not only_image:
            text_optimizer.step()
        train_loss += loss.item()
        num_samples += batch["labels"].size(0)

    train_loss /= num_samples
    return train_loss

def eval(testloader, image_head, text_head, criterion, only_loss=True):
    image_head.eval()
    text_head.eval()
    eval_loss = 0
    num_samples = 0
    image_embeddings = [None for _ in range(testloader.dataset.num_images)]
    text_embeddings = [None for _ in range(testloader.dataset.num_texts)]
    with torch.no_grad():
        for batch in testloader:
            image_embedding = image_head(batch["images"].to(device))
            text_embedding = text_head(batch["texts"].to(device))
            for i in range(len(batch["image_idx"])):
                if image_embeddings[batch["image_idx"][i]] is None:
                    image_embedding_numpy = image_embedding[i].cpu().numpy()
                    image_embeddings[batch["image_idx"][i]] = image_embedding_numpy / np.linalg.norm(image_embedding_numpy, ord=2)
            for i in range(len(batch["text_idx"])):
                if text_embeddings[batch["text_idx"][i]] is None:
                    text_embedding_numpy = text_embedding[i].cpu().numpy()
                    text_embeddings[batch["text_idx"][i]] = text_embedding_numpy / np.linalg.norm(text_embedding_numpy, ord=2)
            loss = criterion(image_embedding, text_embedding, batch["labels"].to(device))
            eval_loss += loss.item()
            num_samples += batch["labels"].size(0)
    eval_loss /= num_samples

    if only_loss:
        sim_mat = None
    else:
        image_embeddings = np.array(image_embeddings)
        text_embeddings = np.array(text_embeddings)
        sim_mat = np.dot(image_embeddings, text_embeddings.T)
    return eval_loss, sim_mat

def eval_decomposed(testloader, image_head, text_head, margin=0.1):
    image_head.eval()
    text_head.eval()
    pos_loss, neg_loss = 0, 0
    pos_samples, neg_samples = 0, 0
    criterion = torch.nn.CosineEmbeddingLoss(margin=margin, reduction='none')
    with torch.no_grad():
        for batch in testloader:
            image_embedding = image_head(batch["images"].to(device))
            text_embedding = text_head(batch["texts"].to(device))
            loss = criterion(image_embedding, text_embedding, batch["labels"].to(device))
            loss = loss.cpu().numpy()
            label = batch["labels"].cpu().numpy()
            pos_loss += np.sum(loss[label == 1])
            neg_loss += np.sum(loss[label == -1])
            pos_samples += len(loss[label == 1])
            neg_samples += len(loss[label == -1])
    pos_loss /= pos_samples
    neg_loss /= neg_samples
    return pos_loss, neg_loss

def proj(dataloader, image_head, text_head, batch_size=32):
    image_head.eval()
    text_head.eval()
    image_num_batches = math.ceil(dataloader.dataset.num_images / batch_size)
    text_num_batches = math.ceil(dataloader.dataset.num_texts / batch_size)
    with torch.no_grad():
        for i in range(image_num_batches):
            if i < image_num_batches - 1:
                image_batch = dataloader.dataset.imagedata_preprocessed[i * batch_size : (i + 1) * batch_size]
            else:
                image_batch = dataloader.dataset.imagedata_preprocessed[i * batch_size :]
            image_batch = torch.FloatTensor(np.array(image_batch)).to(device)
            if i == 0:
                image_embeddings = image_head(image_batch).cpu().numpy()
            else:
                image_embeddings = np.vstack((image_embeddings, image_head(image_batch).cpu().numpy()))
        for i in range(text_num_batches):
            if i < text_num_batches - 1:
                text_batch = dataloader.dataset.textdata_preprocessed[i * batch_size : (i + 1) * batch_size]
            else:
                text_batch = dataloader.dataset.textdata_preprocessed[i * batch_size :]
            text_batch = torch.FloatTensor(np.array(text_batch)).to(device)
            if i == 0:
                text_embeddings = text_head(text_batch).cpu().numpy()
            else:
                text_embeddings = np.vstack((text_embeddings, text_head(text_batch).cpu().numpy()))
    return image_embeddings, text_embeddings

def getMedR(testloader, sim_mat):
    num_images = testloader.dataset.num_images
    image_text_rate = testloader.dataset.image_text_rate
    rank = np.argsort(-sim_mat, axis=0)
    rank = np.argsort(rank, axis=0)
    ranking = [rank[i, i * image_text_rate : (i + 1) * image_text_rate] for i in range(num_images)]
    ranking = np.array(ranking).flatten()
    image_MedR = np.median(ranking)
    rank = np.argsort(-sim_mat, axis=1)
    rank = np.argsort(rank, axis=1)
    ranking = [rank[i, i * image_text_rate : (i + 1) * image_text_rate] for i in range(num_images)]
    ranking = np.array(ranking).flatten()
    text_MedR = np.median(ranking)
    return {"image": image_MedR, "text": text_MedR}

def getRatK(testloader, sim_mat, K=10):
    num_images = testloader.dataset.num_images
    image_text_rate = testloader.dataset.image_text_rate
    rank = np.argsort(-sim_mat, axis=0)
    rank = np.argsort(rank, axis=0)
    ranking = [rank[i, i * image_text_rate : (i + 1) * image_text_rate] for i in range(num_images)]
    ranking = np.array(ranking).flatten()
    image_RatK = sum(ranking < K) / len(ranking)
    rank = np.argsort(-sim_mat, axis=1)
    rank = np.argsort(rank, axis=1)
    ranking = [rank[i, i * image_text_rate : (i + 1) * image_text_rate] for i in range(num_images)]
    ranking = np.array(ranking).flatten()
    text_RatK = sum(ranking < K) / len(ranking)
    return {"image": image_RatK, "text": text_RatK}

def show_performance(testloader, sim_mat, num_samples=50):
    MedR = getMedR(testloader, sim_mat)
    RatK = getRatK(testloader, sim_mat)
    print(f"Image Retrieval From Text:")
    print(f'   Median Rank: {MedR["image"]:.1f} / {testloader.dataset.num_images:.0f}')
    print(f'   Recall at K: {RatK["image"]:.4f}')
    print(f"Text Retrieval From Image:")
    print(f'   Median Rank: {MedR["text"]:.1f} / {testloader.dataset.num_texts:.0f}')
    print(f'   Recall at K: {RatK["text"]:.4f}')

    plt.figure(figsize=(10,8))
    rand_idx = np.random.randint(0, len(sim_mat), num_samples)
    sns.heatmap(sim_mat[rand_idx][:, rand_idx * 10 + np.random.randint(0, 10, num_samples)], cmap="cool", vmin=-1, vmax=1)
    plt.title(f"Similarity Map for the Random {num_samples} Images and Texts")
    plt.xlabel("Text")
    plt.ylabel("Image")
    plt.show()

def TextFromImage(sim_mat, test_images, test_texts, num_image=5, num_text=10):
    image_idx = np.random.randint(0, len(test_images), num_image)
    rank = np.argsort(-sim_mat, axis=1)
    for i in range(num_image):
        print("Query Image:")
        img = test_images[image_idx[i]]
        img = np.flip(img, axis=-1)
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.show()
        for j in range(num_text):
            print(f"CosSim={sim_mat[image_idx[i], rank[image_idx[i], j]]:.4f} | " + test_texts[rank[image_idx[i], j]])
        print("\n")

def ImageFromText(sim_mat, test_images, test_texts, num_text=5, num_image=10):
    text_idx = np.random.randint(0, len(test_texts), num_text)
    rank = np.argsort(-sim_mat, axis=0)
    for i in range(num_text):
        print("Query Text: " + test_texts[text_idx[i]])
        plt.figure(figsize=(20, 5))
        for j in range(num_image):
            plt.subplot(1, num_image, j + 1)
            img = test_images[rank[j, text_idx[i]]]
            img = np.flip(img, axis=-1)
            plt.imshow(img)
            plt.title(f"CosSim={sim_mat[rank[j, text_idx[i]], text_idx[i]]:.4f}")
        plt.tight_layout()
        plt.show()
        print("\n")

def tSNE_visualization(train_image_embeddings, train_text_embeddings, test_image_embeddings, test_text_embeddings, num_samples=10):
    image_text_rate = int(len(train_text_embeddings) / len(train_image_embeddings))
    random_idx = np.random.randint(0, len(train_image_embeddings), num_samples)
    vis_array = np.empty((0, train_image_embeddings.shape[1]))
    label = []
    for idx in random_idx:
        vis_array = np.vstack((vis_array, train_image_embeddings[idx]))
        label.append(idx)
        vis_array = np.vstack((vis_array, train_text_embeddings[idx * image_text_rate : (idx + 1) * image_text_rate]))
        label += ([idx] * image_text_rate)
    label = np.array(label)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_out = tsne.fit_transform(vis_array)
    tsne_df = pd.DataFrame(tsne_out, columns=["x_train", "y_train"])
    tsne_df["label_train"] = label

    random_idx = np.random.randint(0, len(test_image_embeddings), num_samples)
    vis_array = np.empty((0, train_image_embeddings.shape[1]))
    label = []
    for idx in random_idx:
        vis_array = np.vstack((vis_array, test_image_embeddings[idx]))
        label.append(idx)
        vis_array = np.vstack((vis_array, test_text_embeddings[idx * image_text_rate : (idx + 1) * image_text_rate]))
        label += ([idx] * image_text_rate)
    label = np.array(label)
    tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
    tsne_out = tsne.fit_transform(vis_array)
    tsne_df["x_test"] = tsne_out[:, 0]
    tsne_df["y_test"] = tsne_out[:, 1]
    tsne_df["label_test"] = label

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x="x_train", y="y_train", hue="label_train", palette=sns.color_palette("husl", num_samples), data=tsne_df, legend=False, s=80)
    plt.title("t-SNE Visualization on Training Set", fontsize=16)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x="x_test", y="y_test", hue="label_test", palette=sns.color_palette("husl", num_samples), data=tsne_df, legend=False, s=80)
    plt.title("t-SNE Visualization on Test Set", fontsize=16)
    plt.tight_layout()
    plt.show()

def show15samples(dataloader, images, texts):
    label_dic = {1: "positive", -1: "negative"}
    for batch in dataloader:
        plt.figure(figsize=(15, 9))
        for i in range(15):
            if i < len(batch["labels"]):
                plt.subplot(3, 5, i + 1)
                img = images[batch["image_idx"][i]]
                img = np.flip(img, axis=-1)
                plt.imshow(img)
                plt.title("Image: " + str(i + 1))
        plt.tight_layout()
        plt.show()
        for i in range(15):
            if i < len(batch["labels"]):
                label = int(batch["labels"][i].cpu().numpy().tolist())
                print("Text: " + str(i + 1) + " | " + label_dic[label] + " : " + texts[batch["text_idx"][i]])
        break
