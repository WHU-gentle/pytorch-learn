import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Model
from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(args)

# 训练函数
def train(dataset, model, args):
    model.train()  # ???
    # 准备数据
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # 离散分类问题  交叉熵计算损失
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.max_epochs):
        optimizer.zero_grad()
        # 前向推导 计算损失
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        loss = criterion(y_pred.transpose(1, 2), y)  # ???

        state_h = state_h.detach()  # ???
        state_c = state_c.detach()  
        # 误差反向传播
        loss.backward()
        optimizer.step()

        print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})

def predict(dataset, model, text, next_words = 100):
    model.eval()  # ???

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([dataset.word_to_index[w] for w in words[i:]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]  # ?
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach()  # ?
        word_index = np.random.choice(len(last_word_logits), p=p)  # ???
        words.append(dataset.index_to_word[word_index])

    return words

train(dataset, model, args)
print(predict(dataset, model, text='Knock knock. Whos there?'))
