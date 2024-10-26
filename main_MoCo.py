import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.utils as utils
from utils.baseModels import baseMLP


# train for one epoch to learn unique features
def train(encoder_q, encoder_k, data_loader, train_optimizer, device):
    global memory_queue
    encoder_q.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, desc='Training')
    for x_q, x_k, _ in train_bar:
        x_q, x_k = x_q.to(device, non_blocking=True), x_k.to(device, non_blocking=True)
        _, query = encoder_q(x_q)  # (batch_size, feature_dim)

        # shuffle BN
        idx = torch.randperm(x_k.size(0), device=x_k.device)
        _, key = encoder_k(x_k[idx])  # will compute the mean and variance, and the learnable parameters based on this shuffled batch.
        key = key[torch.argsort(idx)]  # (batch_size, feature_dim), BP corresponds to the original order
        
        # TODO: YX: add more positive pairs in the dictionary, i.e., the memory queue, see open thought 1 in the blog.
        
        # Notice that all the features are normalized, so the dot product is cosine similarity!
        # positive logits: Nx1
        score_pos = torch.bmm(
            query.unsqueeze(dim=1),  # (batch_size, 1, feature_dim)
            key.unsqueeze(dim=-1),  # (batch_size, feature_dim, 1)
            ).squeeze(dim=-1)  # (batch_size, 1, 1) > (batch_size, 1)

        # negative logits: NxM
        score_neg = torch.mm(
            query,  # (batch_size, feature_dim)
            memory_queue.t().contiguous()  # (feature_dim, M)
            )  # (batch_size, M)

        out = torch.cat([score_pos, score_neg], dim=-1)  # (batch_size, 1+M)
        # compute loss
        loss = F.cross_entropy(
            out / temperature,  # (batch_size, 1+M)
            torch.zeros(x_q.size(0), dtype=torch.long, device=x_q.device)  # (batch_size, ),all zeros, indicating the positive class.
            )

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        # momentum update
        for parameter_q, parameter_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * momentum + parameter_q.data * (1.0 - momentum))

        # update queue, queue size is (M, feature_dim)
        memory_queue = torch.cat((memory_queue, key), dim=0)[key.size(0):]

        total_num += x_q.size(0)
        total_loss += loss.item() * x_q.size(0)
        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, device):
    net.eval()  # model_q
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data = data.to(device, non_blocking=True)
            feature, out = net(data)
            feature_bank.append(feature)
        
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)

        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='Testing')
        for data, _, target in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank --> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)  # k most similar samples 
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)  # collects the labels from feature_labels at the positions specified by sim_indices, now contains the labels of the k most similar samples for each feature vector in the batch.
            sim_weight = (sim_weight / 0.5).exp()  # temperature= 0.5

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score --> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)

            # top-1 accuracy
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # top-5 accuracy
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            test_bar.set_description(f'Test Epoch: [{epoch}/{epochs}] Acc@1:{total_top1 / total_num * 100:.2f}% Acc@5:{total_top5 / total_num * 100:.2f}%')

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOCO based Traffic Anomaly Detection')
    parser.add_argument('--feature_dim', default=64, type=int, help='Feature dim for each image')
    parser.add_argument('--m', default=2048, type=int, help='Negative sample number')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in softmax')
    parser.add_argument('--momentum', default=0.99, type=float, help='Momentum used for the update of memory bank')
    parser.add_argument('--k', default=6, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=800, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, m, temperature, momentum = args.feature_dim, args.m, args.temperature, args.momentum
    k, batch_size, epochs = args.k, args.batch_size, args.epochs

    # train data prepare
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])  # number of workers

    train_data = utils.TrafficDataset(root='./data/Kowloon_Final_training.csv')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
    
    # test data
    memory_data = utils.TrafficDataset(root='./data/Kowloon_Final_training.csv')
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_data = utils.TrafficDataset(root='data/Kowloon_Final_testing.csv')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # c as num of train class
    c = len(memory_data.classes)

    # model setup and optimizer config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_q = baseMLP(in_dim=720, feature_dim=feature_dim).to(device)
    model_k = baseMLP(in_dim=720, feature_dim=feature_dim).to(device)

    # initialize
    for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
        param_k.data.copy_(param_q.data)
        # not update by gradient
        param_k.requires_grad = False
    
    # only update the parameters in model_q
    optimizer = torch.optim.Adam(model_q.parameters(), lr=1e-3, weight_decay=1e-6)

    # init memory queue as unit random vector ---> [M, D]
    memory_queue = F.normalize(torch.randn(m, feature_dim).to(device), dim=-1)  # L2 normalization by default

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = f'{feature_dim}_{m}_{temperature}_{momentum}_{k}_{batch_size}_{epochs}'
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model_q, model_k, train_loader, optimizer, device)
        results['train_loss'].append(train_loss)

        # test
        test_acc_1, test_acc_5 = test(model_q, memory_loader, test_loader, device)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        # save statistics
        os.makedirs('res/res_CL', exist_ok=True)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'res/res_CL/{save_name_pre}_results.csv', index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model_q.state_dict(), f'res/res_CL/{save_name_pre}_model.pth')