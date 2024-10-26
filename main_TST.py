import pandas as pd
from tsai.all import *
import argparse
import torch
import torch.nn as nn
from fastai.metrics import *

# my_setup()
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load data
def get_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2].values
    splits = get_splits(y, valid_size=878, random_state=42, shuffle=True, stratify=True, train_only=False, show_plot=False)

    # for test:
    # print(f'Data shape: {X.shape}, target shape: {y.shape}')
    # print(f'Splits: {splits}')
    # splits

    tfms = [None, [Categorize()]]
    tsds = TSDatasets(X, y, splits=splits, tfms=tfms, inplace=True)  # inplace=True: The transformations are applied directly to the original dataset. This means that the original data will be modified.

    return tsds


# Metrics
def top_5p_accuracy(inp, targ):
    return top_k_accuracy(inp, targ, k=max(1, inp.shape[-1] // 20))

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='TrafficAnomalyTransformer')
    parser.add_argument('--epochs', default=200, type=int, help='The epoch limit (default: 300)')
    parser.add_argument('--batch_size', default=256, type=int, help='The batch size (default: 64)')
    parser.add_argument('--lr', default=1e-3, type=float, help='The learning rate (default: 0.001)')
    # parser.add_argument('--c_out', default=32, type=float, help='The number of output (default: 32)')
    args = parser.parse_args()

    # Load data
    tsds = get_dataset('./data/Kowloon_Data_processed.csv')
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])  # number of workers
    tsdl = TSDataLoaders.from_dsets(tsds.train, tsds.valid, bs=[256, 256], batch_tfms=[TSStandardize()], num_workers=nw)
    # c_in: tsdl.vars, num_targs: tsdl.c, lengths: tsdl.len
    
    # Build model
    model = TST(c_in=tsdl.vars, c_out=tsdl.c, seq_len=tsdl.len, n_layers=6, n_heads=16, dropout=0.3,).to(dev)

    learn = Learner(
        tsdl, 
        model, 
        loss_func=nn.CrossEntropyLoss(), 
        metrics=[accuracy, top_5p_accuracy], 
    )
    learn.fit_one_cycle(args.epochs, lr_max=args.lr,)
    learn.save_all(f'./res/res_TST/')
    learn.plot_metrics()

    # save the plot
    plt.savefig('./res/res_TST/metrics.png')

