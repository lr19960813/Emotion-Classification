import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import clip
from PIL import Image
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
from sklearn import metrics
import matplotlib.pyplot as plt


class Artemis_testing(Dataset):
    def __init__(self): #
        csv = pd.read_csv('D:/Dataset/artemis_dataset/resnet/label_testing2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = 'D:/Dataset/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index): #
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index]+'.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self): #
        return self._len

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    #
    ind_array = np.arange(len(label))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

classes = ['amusement','awe','contentment','excitement','fear','sadness','anger','disgust','somthingelse']
label = ['amusement','awe','contentment','excitement','fear','sadness','anger','disgust','somthingelse']
emotion = ['amused','in awe','content','excited','fearful','sad','angry','disgusted','nothing']
emotiongood = ['amusing','awe','content','thrilled','fear','tragic','irritated','repelled','nought']
emotionnn = ['amused','in awe','content','excited','timid','sad','angry','disgusted','nothing']
emotional = ['amusing','awesome','satisfactory','exciting','fearful','sad','angry','disgusting','simple'] #A emotional painting


convertor = Artemis_testing() #
BATCH_SIZE = 8
test_dataloader = DataLoader(convertor,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)

#Load the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

checkpoint = torch.load("D:/Dataset/clip_model/exp_lr/clip5e-06_0.8.pt") #Load the fine tuned CLIP
model.load_state_dict(checkpoint['model_state_dict'])

if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

# Prepare the inputs
with torch.no_grad():
    total, correct, count = 0,0,0
    predicted_final = np.zeros(shape=(0))
    labels_final = np.zeros(shape=(0))
    for data in test_dataloader:
        list_image, list_emotion, list_repetition = data
        images = list_image.cuda() #Input image
        texts = clip.tokenize(f"This painting causes me feel {c}." for c in emotion).to(device) #Design Prompt
        logits_per_image, logits_per_text = model(images, texts) #Calculate Similarity between one image with 9 texts.
        _, predicted = torch.max(logits_per_image.data, dim=1)
        labels = torch.from_numpy(np.asarray(list_emotion).astype(float)).to(device)

        predicted_np = predicted.cpu().numpy()
        labels_np = labels.cpu().numpy()
        labels_np = labels_np.astype(np.int)
        predicted_final = np.append(predicted_final, predicted_np)
        labels_final = np.append(labels_final, labels_np).astype(np.int)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count += 1
        if count % 20 == 19:
            print(f"top1: {100 * correct/total}%")
    print(f'Classification emotion Accuracy on testset {100*correct/total}% test images:{total}',)
    
    # Get F1 score
    print(metrics.confusion_matrix(labels_final, predicted_final))
    print(metrics.classification_report(labels_final, predicted_final, digits=9))
    # Confusion matrix
    cm = confusion_matrix(labels_final, predicted_final)
    print(cm)
    # Normalize by row
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized, 'confusion_matrix.png', title='confusion matrix')

