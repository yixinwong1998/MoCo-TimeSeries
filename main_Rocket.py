import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tsai.all import *
from sklearn.metrics import log_loss


# my_setup()
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load data
def get_dataset(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-2].values
    y = df.iloc[:, -2].values
    splits = get_splits(y, valid_size=878, random_state=42, shuffle=True, stratify=True, train_only=False)

    # for test
    # print(f'Data shape: {X.shape}, target shape: {y.shape}')
    # print(f'Splits: {splits}')
    # splits

    tfms = [None, [Categorize()]]
    tsds = TSDatasets(X, y, splits=splits, tfms=tfms, inplace=True)  # inplace=True: The transformations are applied directly to the original dataset. This means that the original data will be modified.

    return tsds

if __name__ == '__main__':
    # Load data
    tsds = get_dataset('./data/Kowloon_Data_processed.csv')
    nw = 4  # min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 16])  # number of workers
    tsdl = TSDataLoaders.from_dsets(tsds.train, tsds.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=nw)

    # Build model
    model = build_ts_model(ROCKET, dls=tsdl, device=dev, arch_config={'n_kernels': 2000})
    X_train, y_train = create_rocket_features(tsdl.train, model)
    X_valid, y_valid = create_rocket_features(tsdl.valid, model)

    # Apply a classifier
    clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    train_losses = []
    valid_losses = []

    # Fit the model and calculate losses
    clf.fit(X_train, y_train)
    train_pred_proba = clf.predict_proba(X_train)
    valid_pred_proba = clf.predict_proba(X_valid)

    train_loss = log_loss(y_train, train_pred_proba)
    valid_loss = log_loss(y_valid, valid_pred_proba)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # Print accuracy
    accuracy = clf.score(X_valid, y_valid)
    print(f'Accuracy: {accuracy}')

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()






