import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import random
from collections import OrderedDict


from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *




def features_org(seq):
    train_Xs = []
    train_Ys = []
    train_Us = []
    test_Xs = []
    test_Ys = []
    test_Us = []
    train_len0 = 0
    train_len1 = 0
    test_len0 = 0
    test_len1 = 0
    for item in seq:
        for features in seq[item]:

            # if random.random() >0.2: # downsample
            #     continue

            if random.random() < 0.9:
                if features[-1] == 1 and train_len1 > 1.2 * train_len0:
                    continue
                if features[-1] == 1:
                    train_len1 += 1
                else:
                    train_len0 += 1
                train_Us.append(features[:13])
                train_Xs.append([features[13+9*i: 13+9*i+9] for i in range(4)])
                train_Ys.append([features[-1]])
            else:
                if features[-1] == 1 and test_len1 > 1.2 * test_len0:
                    continue
                if features[-1] == 1:
                    test_len1 += 1
                else:
                    test_len0 += 1
                test_Us.append(features[:13])
                test_Xs.append([features[13+9*i: 13+9*i+9] for i in range(4)])
                test_Ys.append([features[-1]])
    print(len(test_Us))
    print(len(test_Xs))
    print(len(test_Ys))
    print(len(train_Us))
    print(len(train_Xs))
    print(len(train_Ys))

    return (torch.tensor(train_Us), torch.tensor(train_Xs), torch.tensor(train_Ys), torch.tensor(test_Us), torch.tensor(test_Xs), torch.tensor(test_Ys))


class Seq2SeqDataset(Dataset):

    def __init__(self, U, X, Y):
        super().__init__()

        # check inputs
        assert len(X) == len(Y) and len(X) == len(U) and len(X) > 0

        self.Us = U
        self.Xs = X
        self.Ys = Y

        self.num_example = len(U)

    def __getitem__(self, idx):
        return self.Us[idx], self.Xs[idx], self.Ys[idx]

    def __len__(self):
        return self.num_example


class RNN(nn.Module):
    """ RNN Encoder-decoder model
    """
    def __init__(self, enc_input_dim, output_dim, user_dim, hidden_dim, layers, bid, user):
        super().__init__()
        # encoder
        self.norm1 = nn.BatchNorm1d(enc_input_dim)
        self.norm2 = nn.BatchNorm1d(user_dim)
        self.enc_rnn = nn.GRU(
            input_size=enc_input_dim,
            hidden_size=hidden_dim,
            num_layers= layers,
            batch_first=True,
            bidirectional=bid
        )

        self.bid = bid
        self.user = user
        if bid:
            linear_dim = hidden_dim * 2 * layers + user
        else:
            linear_dim = hidden_dim * layers + user

        # linear predictor
        self.linear1 = nn.Linear(linear_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)
        self.pre = nn.Sigmoid()

    def forward(self, X, U):
        X = self.norm1(X.transpose(1, 2)).transpose(1, 2)
        U = self.norm2(U)
        output, hidden = self.enc_rnn(X)  # encoding

        if self.bid:
            hidden = torch.cat([hidden[i] for i in range(len(hidden))], 1).unsqueeze(0)
        if self.user > 0:
            hidden = torch.cat([hidden, U.unsqueeze(0)], 2)

        # print(hidden)
        y_hat = self.linear2(self.linear1(hidden))  # predicting
        y_hat = self.pre(y_hat)
        # print(y_hat)
        return y_hat

def loss_count(pred, true):
    count = 0
    for i in range(len(pred)):
        if (pred[i][0] >= 0.5 and true[i][0] == 1) or (pred[i][0] < 0.5 and true[i][0] == 0):
            count += 1
    return count / len(true)

def train(features_seq):

    input_dim = 9
    output_dim = 1
    hidden_dim = 10
    batch_size = 32
    user_dim = 13
    loss_fn = nn.BCELoss()
    learning_rate = 0.05

    (train_Us, train_Xs, train_Ys, test_Us, test_Xs, test_Ys) = features_org(features_seq)

    train_dataset = Seq2SeqDataset(train_Us, train_Xs, train_Ys)
    test_dataset = Seq2SeqDataset(test_Us, test_Xs, test_Ys)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size)


    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)
    param_tensor = torch.zeros([1], requires_grad=True)
    default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)
    def get_default_trainer():
        def train_step(engine, batch):
            return batch
        return Engine(train_step)
    # create default model for doctests
    default_model = nn.Sequential(OrderedDict([
        ('base', nn.Linear(4, 2)),
        ('fc', nn.Linear(2, 1))
    ]))
    manual_seed(666)

    roc_auc = ROC_AUC()
    roc_auc.attach(default_evaluator, 'roc_auc')

    history = []
    model = RNN(input_dim, output_dim, user_dim, hidden_dim, 2, True, len(train_Us[0]))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.double().cuda()

    best_score = 0
    for epoch in range(100):
        # training loop
        model.train()
        losses = []
        for i, batch in enumerate(train_loader):
            # forward pass
            U, X, Y = batch
            U, X, Y = U.cuda(), X.cuda(), Y.cuda()
            y_hat = model(X, U)

            loss = loss_fn(y_hat.double().squeeze(0), Y.double())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach())

        # calculate epoch loss and MAPE
        train_loss = torch.mean(torch.tensor(losses)).item()

        # evaluation loop
        model.eval()
        losses = []
        loss_num = []
        auc_list = []
        with torch.no_grad():
            for batch in eval_loader:
                # forward pass
                U, X, Y = batch
                U, X, Y = U.cuda(), X.cuda(), Y.cuda()
                y_hat = model(X, U)

                loss = loss_fn(y_hat.double().squeeze(0), Y.double())

                loss_num.append(loss_count(y_hat.double().squeeze(0), Y.double()))
                losses.append(loss.detach())
                auc_list.append(default_evaluator.run([[y_hat.squeeze(0), Y]]).metrics['roc_auc'])
            # calculate evaluateion loss and MAPE for current epoch
            auc_loss = torch.mean(torch.tensor(auc_list)).item()
            eval_loss = torch.mean(torch.tensor(losses)).item()
            count_loss = torch.mean(torch.tensor(loss_num)).item()
            if count_loss > best_score:
                best_score = count_loss
                best_ckpt = model.state_dict()
        # record training curves and metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'pred': count_loss,
            'AUC': auc_loss,
        }
        history.append(metrics)

        print(metrics)

    print(best_score)
    torch.save({
        'model_state_dict': best_ckpt,
    }, "./ckpt/checkpoint.ckpt")






