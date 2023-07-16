import torch
import numpy as np
import pandas as pd
from PIL import Image

import clip
from clip import model

from Attention_visualizer.attention_visualizer import *

def find_best_matches(text_features, photo_features, photo_ids, results_count):
  similarities = (photo_features @ text_features.T).squeeze(1)
  best_photo_idx = (-similarities).argsort()
  return [photo_ids[i] for i in best_photo_idx[:results_count]]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

photo_ids = pd.read_csv("unsplash-dataset/photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])
photo_features = np.load("unsplash-dataset/features.npy")

search_query = "A red flower is under the blue sky and there is a bee on the flower"

with torch.no_grad():
    text_token = clip.tokenize(search_query).to(device)
    text_encoded, weight = model.encode_text(text_token)
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

text_features = text_encoded.cpu().numpy()
best_photo_ids = find_best_matches(text_features, photo_features, photo_ids, 5)

for photo_id in best_photo_ids:
  print("https://unsplash.com/photos/{}/download".format(photo_id))

sentence = search_query.split(" ")
attention_weights = list(weight[-1][0][1+len(sentence)].cpu().numpy())[:2+len(sentence)][1:][:-1]
attention_weights = [float(item) for item in attention_weights]
display_attention(sentence,attention_weights)