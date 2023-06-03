import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from torch import nn
from torch.utils import data
from torch.nn import functional as F

def load_array(image, label, batch_size, is_train=True):
    dataset = data.TensorDataset(torch.Tensor(image), torch.tensor(label, dtype=torch.int64))
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

class MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        logits = self.linear_layer(x)
        return logits

class CNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(1,32,3,padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(32,64,3,padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding='valid'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten=nn.Flatten()
        self.linear=nn.Sequential(
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        x=torch.reshape(x,(-1,1,28,28))
        x=self.block1(x)
        x=self.block2(x)
        x=self.flatten(x)
        x=self.linear(x)
        return x

def weights_init(m):
    if type(m) == nn.Linear:
        # nn.init.normal_(m.weight,0,1/np.sqrt(m.in_features))
        # nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def evaluate(model, data_iter):
    model.eval()
    total_num = 0
    acc_num = 0
    with torch.no_grad():
        for data in data_iter:
            image,label=data[0].cuda(),data[1].cuda()
            label_hat = model(image).argmax(axis=1)
            cmp = label_hat.type(label.dtype) == label
            acc_num += cmp.type(label.dtype).sum()
            total_num += label.numel()
    return float(acc_num)/total_num

def plot_training(loss,scores,fpath):
    fig,ax1=plt.subplots()
    ax1.plot(loss,label='loss',color='red')
    ax1.legend(loc='upper left')
    ax2=ax1.twinx()
    ax2.plot(scores,label='valid_acc',color='blue')
    ax2.legend(loc='upper right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('valid_acc')
    ax1.set_title('Training process')
    plt.savefig(fpath)


def train(model, train_iter, valid_iter, loss, epochs, optimizer):
    running_loss = 0.
    count = 0
    losses=[]
    scores=[]
    patient,best_acc=0,0.0
    for epoch in range(epochs):
        model.train()
        for data in train_iter:
            image,label=data[0].cuda(),data[1].cuda()
            label_hat = model(image)
            l = loss(label_hat, label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            running_loss += l.item()
            count += 1
        valid_acc=evaluate(model,valid_iter)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model=model
            # torch.save(model,'model.pt')
            # patient=0
        # else:
        #     patient+=1
        #     if patient>10:
        #         break
        print(f'[epoch {epoch}] loss:{running_loss/count}, valid_acc:{valid_acc}')
        losses.append(running_loss/count)
        scores.append(valid_acc)
    # plot_training(losses,scores,'training.png')
    return losses,scores,best_model

# define settings
parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=50,
                    help='number of classes used')
parser.add_argument('--num_samples_train', type=int, default=15,
                    help='number of samples per class used for training')
parser.add_argument('--num_samples_test', type=int, default=5,
                    help='number of samples per class used for testing')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--model',type=str,default='cnn',
                    help='type od the model')
parser.add_argument('--split',type=float,default='0.2',
                    help='train dataset split ratio')
parser.add_argument('--bsz',type=int,default=32,
                    help='batch size')
parser.add_argument('--lr',type=float,default=0.1,
                    help='learning rate')
parser.add_argument('--path',type=str,default='./',
                    help='outfile path')
args = parser.parse_args()

if args.model=='mlp':
    model = MLP(num_classes=args.num_classes)
elif args.model=='cnn':
    model=CNN(num_classes=args.num_classes)

model.apply(weights_init)
print(model)
model.cuda()

batch_size, epochs, lr = args.bsz, 100, args.lr

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

train_image, train_label, test_image, test_label = LoadData(args.num_classes, args.num_samples_train, args.num_samples_test, args.seed)
train_image = np.where(train_image == 1, 1, 0)
test_image = np.where(test_image == 1, 1, 0)
val_size=int(args.split*len(train_image))
dataset = data.TensorDataset(torch.Tensor(train_image), torch.tensor(train_label, dtype=torch.int64))
train_data,valid_data = data.random_split(dataset,[len(dataset)-val_size,val_size])
train_iter=data.DataLoader(train_data, batch_size, shuffle=True)
valid_iter=data.DataLoader(valid_data, batch_size, shuffle=False)
# train_iter = load_array(train_image, train_label,
                        # batch_size=batch_size, is_train=True)
test_iter = load_array(test_image, test_label,
                       batch_size=batch_size, is_train=False)


losses,scores,best_model=train(model, train_iter, valid_iter, loss, epochs, optimizer)
plot_training(losses,scores,args.path+'/training.png')
print(evaluate(best_model,test_iter))