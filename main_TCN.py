import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.TCN import TemporalConvNet

from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrafficAnomalyTCN(nn.Module):
    def __init__(self, 
                num_input_channel,  # univariate time series is 1, multivariate time series is > 1
                length,  # the length of the time series
                mlp_nn,  # the number of nn of MLP layer
                output_size,  # the classes if the model is classification
                num_channels,  # hidden_channel at each layer. eg. [24,12,3] represents 3 blocks in TCN.
                kernel_size,  # the kernel size of the TCN 
                dropout  # drop_out rate
                ):
        super(TrafficAnomalyTCN, self).__init__()

        self.tcn = TemporalConvNet(num_input_channel, num_channels, kernel_size=kernel_size, dropout=dropout)
        # MLP for encoding: input_size -> output_size
        self.linear1 = nn.Linear(length, mlp_nn)
        self.activation = nn.LeakyReLU()
        self.linear2 = nn.Linear(mlp_nn, output_size)

        # Initialize weights
        self.init_weights()
    def init_weights(self):
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='leaky_relu')
    
    def forward(self, x):
        # Convert the input shape: (batch_size, seq_len) to (batch_size, channels, seq_len) first.
        # x.unsqueeze(1) (batch_size, seq_len) --> (batch_size, 1, seq_len)
        x = self.tcn(x.unsqueeze(1))  # --> (batch_size, 1, seq_len)
        x = self.linear1(x.squeeze(1))  # --> (batch_size, mlp_nn)
        logits = self.linear2(x)  # --> (batch_size, output_size), here output_size is the number of classes
        logits = self.linear2(self.activation(x))  # --> (batch_size, output_size)  add the activation layer
        pred = F.log_softmax(logits, dim=1)

        return pred

def train(model, train_data_loader, train_optimizer, clip_rate):
    model.train()
    total_train_loss, total_num = 0, 0
    training_bar = tqdm(train_data_loader, desc='Training')
    for x, y in training_bar:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        pred = model(x)
        loss = F.cross_entropy(pred, y)
        train_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_rate)

        train_optimizer.step()
        total_train_loss += loss.item()
        total_num += x.size(0)
        total_train_loss += loss.item()
        training_bar.set_description(f'Training Loss: {total_train_loss / total_num:.4f}')

    return total_train_loss / total_num

def test(model, test_data_loader):
    model.eval()
    total_test_loss, total_top1, total_top5, total_num = 0, 0, 0, 0
    with torch.no_grad():
        for _, (x, y) in enumerate(test_data_loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            pred = model(x)
            loss = F.cross_entropy(pred, y)
            total_num += x.size(0)
            total_test_loss += loss.item()
            # accuracy
            num_classes = pred.size(1)
            top1_percent_k = max(1, num_classes // 100)
            top5_percent_k = max(1, num_classes // 20)
            total_top1 += torch.sum(pred.topk(top1_percent_k, dim=1)[1] == y.unsqueeze(1)).item()
            total_top5 += torch.sum(pred.topk(top5_percent_k, dim=1)[1] == y.unsqueeze(1)).item()
        
    return total_test_loss / total_num, total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='TrafficAnomalyTCN')
    parser.add_argument('--epochs', default=150, type=int, help='The epoch limit (default: 300)')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size (default: 64)')
    parser.add_argument('--lr', default=1e-3, type=float, help='The learning rate (default: 0.001)')
    parser.add_argument('--mlp_nn', default=64, type=list, help='The number of nn of MLP layer')
    parser.add_argument('--kernel_size', default=5, type=int, help='The kernel size of the TCN')
    parser.add_argument('--dropout', default=0.2, type=float, help='The drop_out rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='The weight decay (default: 1e-5)')  # L2 regularization
    parser.add_argument('--clip', default=0.6, type=float, help='The gradient clipping rate (default: 0.6)')
    # parser.add_argument('--output_size', default=64, type=int, help='The output_size of the model, the number of classes if the model is classification')
    args = parser.parse_args()
    num_channels = [24, 16, 4, 1]  # hidden_channel at each layer. eg. [24,12,3] represents 3 blocks in TCN.

    # Load data
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])  # number of workers
    train_data = utils.TrafficDataset2(root='./data/Kowloon_Final_training.csv')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=nw)
    test_data = utils.TrafficDataset2(root='./data/Kowloon_Final_testing.csv')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    # Model
    model = TrafficAnomalyTCN(
        num_input_channel=1,  # univariate time series is 1, multivariate time series is > 1
        length=train_data.features.shape[1],  # the length of the time series 
        mlp_nn=args.mlp_nn, 
        output_size=len(train_data.classes), 
        num_channels=num_channels, 
        kernel_size=args.kernel_size, 
        dropout=args.dropout,
    ).to(dev, non_blocking=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    # Train loop
    results = {'train_loss': [], 'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    model_name = f'TCN_kernels_{"_".join(map(str, num_channels))}'
    training_name = f'lr{args.lr}_batch{args.batch_size}_mlp_nn{args.mlp_nn}_kernel_size{args.kernel_size}_dropout{args.dropout}'
    os.makedirs(f'./res/res_TCN/{model_name}/', exist_ok=True)
    best_acc = 0.0
    for epoch in tqdm(range(1, args.epochs + 1), desc='Epochs'):
        # Train
        train_loss = train(model, train_loader, optimizer, clip_rate=args.clip)
        scheduler.step()
        results['train_loss'].append(train_loss)
        # Test
        test_loss, test_acc1, test_acc5 = test(model, test_loader)
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc1)
        results['test_acc@5'].append(test_acc5)
        tqdm.write(f'Epoch: {epoch}/{args.epochs} Train Loss: {train_loss:.4f} Test Loss: {test_loss:.4f} Test Acc@1: {test_acc1:.2f}% Test Acc@5: {test_acc5:.2f}%')
        # save best model
        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(model.state_dict(), f'./res_TCN/{model_name}/best_model_{training_name}.pth')

        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'./res_TCN/{model_name}/results_{training_name}.csv', index_label='epoch')
