#!/usr/bin/env python3
# -*- coding: utf-8 -*
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import clip
import torch.nn as nn
import torch
import torch.optim as optim
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import os

class Artemis_training(Dataset):
    def __init__(self):  #
        csv = pd.read_csv('/beegfs/rui/artemis_dataset/artemis_training.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.description = csv['utterance']
        self.file_name = csv['painting']
        self._len = len(self.emotion)
        self.painting_root = '/beegfs/rui/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  #
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        description = self.description[index]
        emotion = self.emotion[index]
        return image, emotion, description

    def __len__(self):  #
        return self._len

class Artemis_validation(Dataset):
    def __init__(self):  #
        csv = pd.read_csv('/beegfs/rui/artemis_dataset/resnet/label_validation.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/beegfs/rui/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  #
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  #
        return self._len

class Artemis_testing(Dataset):
    def __init__(self):  #
        csv = pd.read_csv('/beegfs/rui/artemis_dataset/resnet/label_testing.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/beegfs/rui/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  #
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  #
        return self._len


def main():
    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    BATCH_SIZE = 100
    lr = 1e-5
    EPOCH = 30
    GAMMA = 0.95
    # Define your own dataloader
    convertor_train = Artemis_training()
    convertor_validation = Artemis_validation()
    train_num = len(convertor_train)
    val_num = len(convertor_validation)

    train_loader = DataLoader(convertor_train, batch_size=BATCH_SIZE, drop_last=True,shuffle=True)
    validation_loader = DataLoader(convertor_validation,batch_size=BATCH_SIZE,drop_last=False,shuffle=True)

    # https://github.com/openai/CLIP/issues/57
    def convert_models_to_fp32(model):
        for p in model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training

    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

    # Loss function
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6,weight_decay=0.2)
    
    # learning rate scheduler
    scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = GAMMA)
    lr_exp = []

    # Training start
    loss = []
    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(EPOCH):
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            optimizer.zero_grad()
            list_image, emotion, list_txt = data  # list_images is list of image in numpy array(np.uint8), or list of PIL images

            images = list_image.cuda()  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
            texts = clip.tokenize(list_txt).to(device)

            logits_per_image, logits_per_text = model(images, texts) #calculate sim(innder product between images and captions)

            if device == "cpu":
                ground_truth = torch.arange(BATCH_SIZE).long().to(device)
            else:
                ground_truth = torch.arange(BATCH_SIZE).long().to(device)

            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

            running_loss += total_loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, EPOCH, total_loss)

        # validation
        acc = 0.0
        emotion = ['amusement', 'awe', 'contentment', 'excitement', 'fear', 'sad', 'angry', 'disgust', 'something else']
        with torch.no_grad():
            val_bar = tqdm(validation_loader)
            for val_data in val_bar:
                val_images, val_labels, val_repetition = val_data
                images = val_images.cuda()
                texts = clip.tokenize(f"This painting makes me feel {c}." for c in emotion).to(device) # Design input text, prompt
                logits_per_image, logits_per_text = model(images, texts) # Calculate which text is most similar to this image.
                _, predicted = torch.max(logits_per_image.data, dim=1)
                labels = torch.from_numpy(np.asarray(val_labels).astype(float)).to(device)

                acc += torch.eq(predicted, labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch+1, EPOCH)

        # update learning rate
        lr_exp += scheduler_exp.get_lr()
        scheduler_exp.step()
        # Draw figure
        loss_value2 = total_loss.cpu().detach().numpy().tolist()
        loss.append(loss_value2)

        # print(f'Classification emotion Accuracy on testset {100 * correct / val_num}% test images:{total}', )
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss, }, f'/home/rui/timecompare/clip{lr}_{GAMMA}_{epoch}.pt')
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss, }, f'/home/rui/timecompare/clip{lr}_{GAMMA}_{epoch}.pt')

    x1 = np.linspace(0, EPOCH, EPOCH, dtype=float)
    plt.rcParams['axes.unicode_minus'] = False  #
    plt.ion()  #
    plt.plot(x1, loss, color='green', label='total_loss', lw=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.grid()
    plt.title('Loss/Accuarcy', fontsize=16)
    plt.legend('lower upper')
    plt.savefig(f"{lr}loss_{GAMMA}.jpg", dpi=300)

    print('Finished Training')

main()







