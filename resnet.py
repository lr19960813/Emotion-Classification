import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.models.resnet
from model import resnet34
import clip
import pandas as pd


class Artemis_training(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_training2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len


class Artemis_validation(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_validation2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len


class Artemis_testing(Dataset):

    def __init__(self):  # 载入数据
        csv = pd.read_csv('/home/lirui/Downloads/code/artemis_dataset/resnet/label_testing2.csv', low_memory=False)
        model, preprocess = clip.load("ViT-B/32", device=0, jit=False)
        self.emotion = csv['emotion']
        self.style = csv['artstyle']
        self.repetition = csv['repetition']
        self.file_name = csv['painting']
        self._len = len(self.repetition)
        self.painting_root = '/home/lirui/Downloads/code/artemis_dataset/wikiart'
        self.preprocess = preprocess

    def __getitem__(self, index):  # 根据索引index返回相应位置的图像及标签
        image_path = os.path.join(self.painting_root, self.style[index], self.file_name[index] + '.jpg')
        image = Image.open(image_path)
        image = self.preprocess(image)
        repetition = self.repetition[index]
        emotion = self.emotion[index]
        return image, emotion, repetition

    def __len__(self):  # 查看数据长度
        return self._len


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    convertor_train = Artemis_training()  # 实例化
    convertor_validate = Artemis_validation()

    train_num = len(convertor_train)

    # {'Amusement':0, 'Awe':1, 'Contentment':2, 'Excitement':3, 'Fear':4,'Sadness':5, 'Anger':6, 'Disgust':7, 'something else':8}

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(convertor_train,
                                               batch_size=batch_size, shuffle=True, drop_last=True,
                                               num_workers=nw)

    val_num = len(convertor_validate)
    validate_loader = torch.utils.data.DataLoader(convertor_validate,
                                                  batch_size=batch_size, shuffle=False, drop_last=True,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34()
    # load pretrain weights
    model_weight_path = "./resnet34_pre.pth"  # save the weight
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 9)  # classification amount
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 20
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()  # control bn
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels, repetition = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels, val_repetition = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    convertor_test = Artemis_testing()
    test_num = len(convertor_test)

    # {'Amusement':0, 'Awe':1, 'Contentment':2, 'Excitement':3, 'Fear':4,'Sadness':5, 'Anger':6, 'Disgust':7, 'something else':8}

    batch_size = 5
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 5])  # 8 number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(convertor_test,
                                              batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=nw)

    print("using {} images for testing.".format(test_num))
    net = resnet34()

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 9)  # classification amount
    net.to(device)

    # load pretrain weights
    model_weight_path = "./resNet34.pth"  # save the weight
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))

    # test
    with torch.no_grad():
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader)
            for test_data in test_bar:
                test_images, test_labels, test_repetition = test_data
                outputs = net(test_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accurate = acc / test_num
        print('test_accuracy: %.3f' %
              (test_accurate))

    print('Finished Testing')


# test()

if __name__ == '__main__':
    main()