import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import prep_data
import time

from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset, DataLoader, TensorDataset


def close_price(returns):
    return np.exp(np.cumsum(returns, axis=1))


def train(
    model, 
    seq_len,
    horizon,
    max_epoch, 
    batch_size, 
    loss=nn.MSELoss(),
    optimizer=optim.Adam, 
    lr=1e-4,
    device="cpu"
):
    _, X, y, scaler = prep_data.log_returns(
        seq_len=seq_len, 
        horizon=horizon
    )

    train_test = {
        "X": {},
        "y": {}
    }
    for X_key, y_key in zip(X, y):
        X[X_key] = prep_data.to_tensor(X[X_key])
        y[y_key] = prep_data.to_tensor(y[y_key])
        train_test["X"][X_key] = {}
        train_test["y"][y_key] = {}
        train_test["X"][X_key]["train"], train_test["X"][X_key]["test"] = prep_data.train_test_split(X[X_key])
        train_test["y"][y_key]["train"], train_test["y"][y_key]["test"] = prep_data.train_test_split(y[y_key])
    
    # Initialize some DataFrames for recording stuff
    train_loss_list = []
    test_loss_list = []
    wall_time_list = []
    proc_time_list = []
    
    df_train = pd.DataFrame(
        {"y": scaler.inverse_fit(
            train_test["y"]["Log Returns"]["train"]
            .squeeze(2)
            .squeeze(1)
            .numpy()
            )
        }
    )
    df_test = pd.DataFrame(
        {"y": scaler.inverse_fit(
            train_test["y"]["Log Returns"]["test"]
            .squeeze(2)
            .squeeze(1)
            .numpy()
            )
        }
    )
    
    loader = DataLoader(
        list(zip(
            train_test["X"]["Log Returns"]["train"], 
            train_test["X"]["Timestamp"]["train"], 
            train_test["y"]["Log Returns"]["train"], 
            train_test["y"]["Timestamp"]["train"],
            )
        ), 
        batch_size=batch_size
    )
    model = model.to(device)
    
    backprop = optimizer(model.parameters(), lr=lr)

    X_test = train_test["X"]["Log Returns"]["test"].to(device)
    y_test = train_test["y"]["Log Returns"]["train"].to(device)

    X_test_time = train_test["X"]["Timestamp"]["test"].to(device)
    y_test_time = train_test["y"]["TImestamp"]["train"].to(device)

    for epoch in range(max_epoch):
        start_time_wall, start_time_proc = time.time(), time.process_time()
        model.train()
        total_loss = 0.
        y_train_pred = torch.Tensor()  # Whole sequence of train dataset prediction
        
        for X_batch, X_batch_time, y_batch, y_batch_time in loader:
            start_time = time.time()
            # Forward pass and backpropagation
            y_pred = model(
                src=X_batch.to(device), 
                tgt=y_batch.to(device), 
                src_time=X_batch_time.to(device), 
                tgt_time=y_batch_time.to(device)
            )  # y_pred: (batch_size, horizon, 1)
            batch_loss = loss(y_pred, y_batch.to(device))
            total_loss += batch_loss
            backprop.zero_grad()
            batch_loss.backward()
            backprop.step()
            # Recording predictions
            y_train_pred = torch.cat(
                (
                    y_train_pred.cpu(), 
                    scaler.inverse_fit(y_pred).squeeze(2).squeeze(1).cpu()
                ), 
                0
            )
            torch.cuda.empty_cache() 
        
        end_time_wall, end_time_proc = time.time(), time.process_time()
        
        wall_time = end_time_wall - start_time_wall
        proc_time = end_time_proc - start_time_proc
        
        wall_time_list.append(wall_time)
        proc_time_list.append(proc_time)

        df_train[f"epoch_{epoch}"] = y_train_pred.detach().numpy()  # Save train predictions per epoch
        torch.cuda.empty_cache()
        
        # Evaluate model
        with torch.no_grad():
            model.eval()
            
            y_test_pred = model(
                src=X_test, 
                tgt=y_test, 
                src_time=X_test_time, 
                tgt_time=y_test_time
            )
            
            df_test[f"epoch_{epoch}"] = scaler.inverse_fit(y_test_pred).cpu().squeeze(2).squeeze(1).detach().numpy()

            test_loss = loss(y_test_pred, y_test)
            torch.cuda.empty_cache() 

        train_loss, test_loss = total_loss.cpu().detach().numpy()/len(loader),  test_loss.cpu().detach().numpy()
        # Append train and test loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f"Epoch {epoch} | train loss {train_loss} | test_loss {test_loss} | wall_time {wall_time} | process_time {proc_time}")

    # print(type(train_loss_list[1]), type(test_loss_list[1]))
    loss_df = pd.DataFrame(
        {
            "train_loss": train_loss_list,
            "test_loss": test_loss_list,
            "proc_time": proc_time_list,
            "wall_time": wall_time_list,  
        }
    )

    return model, loss_df, df_train, df_test, scaler


def train_cumsum(
    model, 
    seq_len,
    horizon,
    max_epoch, 
    batch_size, 
    loss=nn.MSELoss(),
    optimizer=optim.Adam, 
    lr=1e-4,
    device="cpu"
):

    # Train test split
    X_returns, X_price, y_returns, y_price, scaler = prep_data.log_returns_w_close(seq_len=seq_len, horizon=horizon)

    X_train, X_test = prep_data.train_test_split(X_returns)
    y_train, y_test = prep_data.train_test_split(y_returns)

    X_price_train, X_price_test = prep_data.train_test_split(X_price)
    y_price_train, y_price_test = prep_data.train_test_split(y_price)
    
    # Initialize some DataFrames for recording stuff
    train_loss_list = []
    test_loss_list = []
    
    df_returns_train = pd.DataFrame(
        {"y": y_train.squeeze(2).squeeze(1).numpy()}
    )
    df_returns_test = pd.DataFrame(
        {"y": y_test.squeeze(2).squeeze(1).numpy()}
    )

    df_price_train = pd.DataFrame(
        {"y": y_price_train.squeeze(2).squeeze(1).numpy()}
    )
    df_price_test = pd.DataFrame(
        {"y": y_price_test.squeeze(2).squeeze(1).numpy()}
    )
    first_price_test = df_price_test["y"].iloc[0]


    loader = DataLoader(
        list(zip(X_train, X_price_train, y_train, y_price_train)), 
        batch_size=batch_size
    )

    model = model.to(device)
    
    backprop = optimizer(model.parameters(), lr=lr)

    X_test, y_test = X_test.to(device), y_test.to(device)
    X_price_test, y_price_test = X_price_test.to(device), y_price_test.to(device)

    for epoch in range(max_epoch):
        model.train()
        total_loss = 0.
        y_train_pred = torch.Tensor()  # Whole sequence of train dataset prediction
        y_train_price_pred = torch.Tensor()

        last_price = float(1.)
        for X_batch, X_price_batch, y_batch, y_price_batch in loader:
            # Forward pass and backpropagation
            y_pred = model(
                scaler.fit(X_batch).to(device), 
                scaler.fit(y_batch).to(device)
            )  # y_pred: (batch_size, horizon, 1)
            batch_loss = loss(y_pred, y_batch.to(device))

            close_train = close_price(
                        scaler.inverse_fit(y_pred).cpu().detach().numpy()
                    )
            # print(type(close_train))
            y_price_pred = last_price * torch.tensor(
                    close_train,
                    requires_grad=True
                )

            # print(f"y_price_pred {y_price_pred.shape}, y_price_batch {y_price_batch.shape}")            
            batch_loss += loss(y_price_pred.to(device), y_price_batch.to(device))

            total_loss += batch_loss
            backprop.zero_grad()
            batch_loss.backward()
            backprop.step()

            last_price = y_price_batch[-1,0::,0::].values
            print(last_price)

            # print(f"y_price_pred {y_price_pred.shape}, y_pred {y_pred.shape}")

            # Recording predictions
            y_train_price_pred = torch.cat(
                (
                    y_train_price_pred.cpu(),
                    y_price_pred.squeeze(2).squeeze(1).cpu()
                ),
                0
            )
            y_train_pred = torch.cat(
                (
                    y_train_pred.cpu(), 
                    y_pred.squeeze(2).squeeze(1).cpu()
                ), 
                0
            )
            torch.cuda.empty_cache() 

        df_returns_train[f"epoch_{epoch}"] = y_train_pred.detach().numpy(  )# Save train predictions per epoch
        df_price_train[f"epoch_{epoch}"] = y_train_price_pred.detach().numpy()
        torch.cuda.empty_cache()
        
        # Evaluate model
        with torch.no_grad():
            model.eval()
            
            y_test_pred = model(
                scaler.fit(X_test), 
                scaler.fit(y_test)
            )

            y_price_test_pred = torch.Tensor(
                first_price_test*close_price(scaler.inverse_fit(y_test_pred).cpu().detach().numpy())
            ).to(device)
            
            df_price_test[f"epoch_{epoch}"] = y_price_test_pred.cpu().squeeze(2).squeeze(1).detach().numpy()
            df_returns_test[f"epoch_{epoch}"] = y_test_pred.cpu().squeeze(2).squeeze(1).detach().numpy()

            test_loss = loss(y_test_pred, y_test)
            test_loss += loss(y_price_test_pred, y_price_test)
            torch.cuda.empty_cache() 

        train_loss, test_loss = total_loss.cpu().detach().numpy()/len(loader),  test_loss.cpu().detach().numpy()

        # Append train and test loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f"Epoch {epoch} | train loss {train_loss} | test_loss {test_loss}")

    print(type(train_loss_list[1]), type(test_loss_list[1]))
    loss_df = pd.DataFrame(
        {
            "train_loss": train_loss_list,
            "test_loss": test_loss_list    
        }
    )

    return model, loss_df, df_returns_train, df_returns_test, df_price_train, df_price_test

def predict(model
):
    pass