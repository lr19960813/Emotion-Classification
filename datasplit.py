import pandas as pd
import os
from sklearn.utils import shuffle

# input csv data
train = pd.read_csv("E:/Dataset/artemis_dataset/resnet/label_total.csv")

# random the dataset and split with 0.8 0.1 0.1
train = shuffle(train,random_state=1)
length = len(train)
print("总长度:",length)
val_num = int(length*0.05)
test_num = int(length*0.1)

train_p = train[:int(length*0.85)]
val_p = train[int(length*0.85):int(length*0.9)]
test_p = train[int(length*0.9):int(length*1)]

print("训练集长度：",len(train_p))
print("验证集长度：",len(val_p))
print("测试集长度：",len(test_p))

# 如果文件不存在，则生成文件
if not os.path.exists("./kialo_data1"):
    os.mkdir("./kialo_data1")

# save the files
val_p.to_csv("./kialo_data1/dev.csv",index=None)
train_p.to_csv("./kialo_data1/train.csv",index=None)
test_p.to_csv("./kialo_data1/test.csv", index=None)


#mass_label_dataset

def read_first_file():
    csv = pd.read_csv('E:/Dataset/artemis_dataset/artemis_testing.csv', low_memory=False)
    csv.sort_values('painting',inplace=True)

    picture = {} # key value 进行排序

    for line in csv.itertuples(): #按行读取
        artstyle = line[1]
        pic_name = line[2]
        emotion = line[3]
        com = artstyle+' '+pic_name+' '+emotion
        if com in picture.keys():
            picture.update({com: picture.get(com)+1})
        else:
            picture.update({com: 1})

    df = pd.DataFrame().from_dict(picture, orient = 'index') #从字典中取出数据，按行取
    df.to_csv('mid_data.csv')
    print('end')

def read_second_file():
    df = pd.read_csv('./mid_data.csv', low_memory=False)

    dict = {}
    for line in df.itertuples():
        pic_name = line[1].split(' ')[0]+' '+line[1].split(' ')[1]
        count = int(line[2])
        if pic_name in dict.keys():
            dict.update({pic_name: dict.get(pic_name) + count})
        else:
            dict.update({pic_name: count})

    res = {}
    for line in df.itertuples():
        pic_name = line[1].split(' ')[0]+' '+line[1].split(' ')[1]
        emo = line[1].split(' ')[2]
        count = int(line[2])

        if count >= dict.get(pic_name) / 2:
            res.update({(pic_name + ' ' + emo): count})

    result = pd.DataFrame().from_dict(res, orient='index')
    result.to_csv('result.csv')
    print('end')


def get_random_sample(_frac):
    df = pd.read_csv('./result.csv', low_memory=False)
    res = df.sample( frac=_frac)
    res.to_csv('rate.csv')

read_first_file()
read_second_file()
get_random_sample(0.8)

csv = pd.read_csv('E:/Dataset/artemis_dataset/artemis_testing.csv', low_memory=False)
csv.sort_values('painting', inplace=True)