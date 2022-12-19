import numpy as np
import tqdm
import math

import torch
from torch.utils.data import Dataset

import Encoders, Preprocessors

class DualEncoderDataset(Dataset):

    def __init__(self, images, texts, device, image_model, text_model, max_len=70, neg_rate=4, preprocessing_batch_size=32):
        self.num_images = len(images)
        self.num_texts = len(texts)
        self.image_text_rate = int(self.num_texts / self.num_images)
        self.num_negative = int(self.num_texts * neg_rate)
        self.random_map = []
        self.shuffle()

        print("\nPreprocessing image data...")
        self.imagedata_preprocessed = []
        image_encoder = Encoders.Encoder(image_model).to(device)
        image_preprocessor = Preprocessors.Preprocessor(image_model)
        num_batches = math.ceil(self.num_images / preprocessing_batch_size)
        for i in tqdm.tqdm(range(num_batches)):
            if i < num_batches - 1:
                image_batch = images[i * preprocessing_batch_size : (i + 1) * preprocessing_batch_size]
            else:
                image_batch = images[i * preprocessing_batch_size :]
            image_batch = torch.FloatTensor(np.array(image_batch)).to(device)
            self.imagedata_preprocessed += image_encoder(image_preprocessor.process(image_batch)).tolist()
        self.image_encoder_num_parameters = image_encoder.num_parameters
        self.image_encoder_output_size = image_encoder.output_size

        print("Preprocessing text data...")
        self.textdata_preprocessed = []
        text_encoder = Encoders.Encoder(text_model).to(device)
        text_preprocessor = Preprocessors.Preprocessor(text_model, max_len)
        num_batches = math.ceil(self.num_texts / preprocessing_batch_size)
        for i in tqdm.tqdm(range(num_batches)):
            if i < num_batches - 1:
                text_batch = texts[i * preprocessing_batch_size : (i + 1) * preprocessing_batch_size]
            else:
                text_batch = texts[i * preprocessing_batch_size :]
            text_batch, attention_batch = text_preprocessor.process(text_batch)
            text_batch = text_batch.to(device)
            if attention_batch is not None:
                attention_batch = attention_batch.to(device)
                self.textdata_preprocessed += text_encoder(text_batch, attention_batch).tolist()
            else:
                self.textdata_preprocessed += text_encoder(text_batch).tolist()
        self.text_encoder_num_parameters = text_encoder.num_parameters
        self.text_encoder_output_size = text_encoder.output_size

    def shuffle(self):
        self.random_map = np.random.randint(0, self.num_texts - self.image_text_rate, self.num_negative)
        for i in range(self.num_negative):
            if self.random_map[i] >= i % self.num_texts // self.image_text_rate * self.image_text_rate:
                self.random_map[i] += self.image_text_rate

    def set_neg_rate(self, neg_rate):
        self.num_negative = int(self.num_texts * neg_rate)
        self.random_map = []
        self.shuffle()

    def print_num_params(self):
        print(f"\n=== Image Encoder ===")
        print(f"  Pretrained Parameters: {self.image_encoder_num_parameters:,}")
        print(f"\n=== Text Encoder ===")
        print(f"  Pretrained Parameters: {self.text_encoder_num_parameters:,}")

    def __len__(self):
        return self.num_texts + self.num_negative

    def __getitem__(self, sample_idx):
        if torch.is_tensor(sample_idx):
            sample_idx = sample_idx[0].tolist()
        image_idx = sample_idx % self.num_texts // self.image_text_rate
        image = self.imagedata_preprocessed[image_idx]
        image = torch.from_numpy(np.array(image)).to(torch.float32)

        if sample_idx < self.num_texts:
            text_idx = sample_idx
        else:
            text_idx = self.random_map[sample_idx - self.num_texts]
        text = self.textdata_preprocessed[text_idx]
        text = torch.from_numpy(np.array(text)).to(torch.float32)

        label = 1 if sample_idx < self.num_texts else -1
        label = torch.from_numpy(np.array(label)).to(torch.float32)

        sample = {"images": image, "texts": text, "labels": label, "image_idx": image_idx, "text_idx": text_idx}
        return sample
