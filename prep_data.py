import numpy as np
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from scaler import Standard, MinMax, DoubleMinMax, No


def log_returns(
    seq_len: int,
    horizon: int,
    ticker: str = "PSEI.PS",
    start_date: str = "2010-01-01",
    end_date: str = "2020-12-31"
):
    """
    Load the Log Returns dataset to a pd.DataFrame object
    """

    idx = pd.date_range("1990-01-01", "2022-12-31")
    final_idx = pd.date_range(start_date, end_date)

    df = yf.download([ticker], "1990-01-01", "2022-12-31")
    df = df.reindex(idx)
    df.fillna(method="ffill", inplace=True)

    df["Timestamp"] = df.index.astype(int)
    df["Timestamp"] = df["Timestamp"].apply(lambda x: float(x))/(86400*1_000_000_000)

    df["Log Returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["Cumulative"] = np.exp(df["Log Returns"].cumsum())

    df_columns = ["Timestamp", "Log Returns", "Cumulative"]

    scaler = {
        "std": Standard(df),
        "minmax": MinMax(df),
        "minmax2": DoubleMinMax(df),
        "none": No(df)
    }

    # Datasets: DataFrames,
    datasets = dict()
    X = dict()
    y = dict()
    

    for column in df_columns:
        datasets[column] =  pd.concat(
        [
            pd.DataFrame(df[column]).rename(columns={column: f"{i}"}).shift(i)
            for i in range(-horizon, seq_len)
        ], 
        axis=1)
        datasets[column] = datasets[column].reindex(final_idx).dropna()
        X[column] = datasets[column][[f"{i}" for i in range(0,seq_len)]]
        y[column] = datasets[column][[f"{i}" for i in range(-horizon,0)]]

    return datasets, X, y, scaler  # Tensors not yet scaled

def to_tensor(df):
    return torch.Tensor(df.to_numpy()).unsqueeze(2)


def log_returns_w_close(
    seq_len: int,
    horizon: int,
    ticker: str = "PSEI.PS",
    start_date: str = "2000-01-01",
    end_date: str = "2020-12-31"
):
    """
    Load the Log Returns dataset to a pd.DataFrame object
    """
    idx = pd.date_range("1990-01-01", "2022-12-31")
    final_idx = pd.date_range(start_date, end_date)

    df = yf.download([ticker], "1990-01-01", "2022-12-31")
    print(df.head())
    df = df.reindex(idx)
    df.fillna(method="ffill", inplace=True)

    df["Log Returns"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    df["Cumulative"] = np.exp(df["Log Returns"].cumsum())

    returns = pd.DataFrame(df[["Log Returns", "Cumulative"]]) 
    # print(returns.tail())
    # print(returns.shift(1).tail())

    scaler = Scaler(df)

    dataset_df_returns = pd.concat(
        [
            pd.DataFrame(returns["Log Returns"]).rename(columns={"Log Returns": f"{i}"}).shift(i)
            for i in range(-horizon, seq_len)
        ], 
        axis=1)

    dataset_df_price = pd.concat(
        [
            pd.DataFrame(returns["Cumulative"]).rename(columns={"Cumulative": f"{i}"}).shift(i)
            for i in range(-horizon, seq_len)
        ], 
        axis=1)

    dataset_df_returns = dataset_df_returns.reindex(final_idx).dropna()
    dataset_df_price = dataset_df_price.reindex(final_idx).dropna()

    X_returns = dataset_df_returns[[f"{i}" for i in range(0,seq_len)]]
    y_returns = dataset_df_returns[[f"{i}" for i in range(-horizon,0)]]

    X_price = dataset_df_price[[f"{i}" for i in range(0,seq_len)]]
    y_price = dataset_df_price[[f"{i}" for i in range(-horizon,0)]]
    
    return (torch.Tensor(X_returns.to_numpy()).unsqueeze(2), 
    torch.Tensor(X_price.to_numpy()).unsqueeze(2),
    torch.Tensor(y_returns.to_numpy()).unsqueeze(2), 
    torch.Tensor(y_price.to_numpy()).unsqueeze(2),
    scaler)  # Scaler only for returns


def train_test_split(df, split: float = 0.8):
    train_split = int(len(df)*0.8)
    df_train = df[:train_split]
    df_test = df[train_split:]

    return df_train, df_test


def load_dataset(
    df,
    seq_len:int,
    horizon:int,
):
    series = pd.DataFrame(df)
    dataset_df = pd.concat([series.shift(i) for i in range(-horizon, seq_len)])
    return dataset_df


def create_inout_sequences(
    df, 
    seq_len, 
    horizon
):
    """
    Create tensors for the models

    Parameters
    ---
    df: pd.DataFrame
    seq_len: int, input
    horizon: int, output
    """
    inout_seq = []
    L = len(df)
    for i in range(L - horizon):
        train_seq = np.append(df[i : i + seq_len][:-horizon], horizon * [0])
        train_label = df[i : i + seq_len]
        # train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


def prep_data(
    df, 
    seq_len:int, 
    horizon: int = 1, 
    split: float = 0.8
):
    amplitude = df.to_numpy()
    amplitude = amplitude.reshape(-1)
    print(amplitude.shape)

    scaler = StandardScaler()
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    sequence = create_inout_sequences(amplitude, seq_len=seq_len, horizon=horizon)

    train_split = int(amplitude.shape[0] * split)



    train_data = amplitude[:split]
    test_data = amplitude[split:]
 
    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
    print(train_sequence.shape)

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
    print(test_data.shape)

    return train_sequence, test_data


def get_batch(df, i, batch_size):
    seq_len = min(batch_size, len(df) - 1 - i)
    data = df[i : i + seq_len]
    input = torch.stack(
        torch.stack([item[0] for item in data]).chunk(input_window, 1)
    )  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


def train(model, train_df):
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_df) - 1, batch_size)):
        data, targets = get_batch(train_df, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        log_interval = int(len(train_df) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | "
                "lr {:02.6f} | {:5.2f} ms | "
                "loss {:5.5f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_df) // batch_size,
                    scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


plot_counter = 0


def plot_and_loss(eval_model, data_source, epoch, tknip):
    global plot_counter
    eval_model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            # look like the model returns static values for the output window
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(
                    output[-output_window:], target[-output_window:]
                ).item()

            test_result = torch.cat(
                (test_result.to(device), output[-1].view(-1).to(device)), 0
            )  # todo: check this. -> looks good to me
            truth = torch.cat((truth.to(device), target[-1].view(-1).to(device)), 0)

    #             test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
    #             truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

    test_result = test_result.cpu().numpy()
    truth = truth.cpu().numpy()
    len(test_result)
    plot_counter += 1
    plt.plot(test_result, color="red")
    plt.plot(truth[:500], color="blue")
    plt.plot(test_result - truth, color="green")
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    plt.savefig(f"./plots2/transformer-epoch_{plot_counter}_{epoch}_{tknip}.png")

    plt.close()

    return total_loss / i


def predict_future(eval_model, data_source, steps, tknip):
    eval_model.eval()
    total_loss = 0.0
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    _, data = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps, 1):
            input = torch.clone(data[-input_window:])
            input[-output_window:] = 0
            output = eval_model(data[-input_window:])
            data = torch.cat((data, output[-1:]))

    #     data = data.to(device).view(-1)
    data = data.cpu().view(-1)
    plt.plot(data, color="red")
    plt.plot(data[:input_window], color="blue")
    plt.grid(True, which="both")
    plt.axhline(y=0, color="k")
    plt.savefig(f"./plots2/transformer-future_{plot_counter}_{steps}_{tknip}.png")
    plt.close()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    eval_batch_size = 32
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            if calculate_loss_over_all_values:
                total_loss += (
                    len(data[0]) * criterion(output, targets).to(device).item()
                )
            #                 total_loss += len(data[0])* criterion(output, targets).cpu().item()
            else:
                total_loss += (
                    len(data[0])
                    * criterion(output[-output_window:], targets[-output_window:])
                    .to(device)
                    .item()
                )
    #                 total_loss += len(data[0])* criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)
