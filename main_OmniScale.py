import pandas as pd
import os
from tsai.all import *
from tsai.metrics import top_k_lift
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
    num_targs = len(np.unique(y))
    splits = get_splits(y, valid_size=878, random_state=42, shuffle=True, stratify=True, train_only=False, show_plot=False)

    # for test:
    # print(f'Data shape: {X.shape}, target shape: {y.shape}')
    # print(f'Splits: {splits}')
    # splits

    tfms = [None, [Categorize()]]
    tsds = TSDatasets(X, y, splits=splits, tfms=tfms, inplace=True)  # inplace=True: The transformations are applied directly to the original dataset. This means that the original data will be modified.

    # c_in: tsdl.vars, num_targs: tsdl.c, lengths: tsdl.len
    return tsds

# Metrics
def top_5p_accuracy(inp, targ):
    return top_k_accuracy(inp, targ, k=max(1, inp.shape[-1] // 20))

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='TrafficAnomalyOmniScaleCNN')
    parser.add_argument('--epochs', default=2, type=int, help='The epoch limit (default: 300)')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size (default: 64)')
    parser.add_argument('--lr', default=1e-3, type=float, help='The learning rate (default: 0.001)')
    # parser.add_argument('--c_out', default=32, type=float, help='The number of output (default: 32)')
    args = parser.parse_args()

    # Load data
    tsds = get_dataset('./data/Kowloon_Data_processed.csv')
    nw = 4  # min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])  # number of workers
    tsdl = TSDataLoaders.from_dsets(tsds.train, tsds.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=nw)

    # Build model
    model = build_ts_model(OmniScaleCNN, dls=tsdl, device=dev, c_in=tsds.vars, c_out=tsds.c, seq_len=tsds.len)
    learn = Learner(
        tsdl, 
        model, 
        loss_func=nn.CrossEntropyLoss(), 
        metrics=[accuracy, top_5p_accuracy], 
    )
    learn.fit_one_cycle(args.epochs, lr_max=args.lr,)
    learn.save_all(f'./res/res_OmniScale/')
    learn.plot_metrics()

