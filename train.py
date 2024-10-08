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

from scaler import Scaler

def close_price(returns):
    return np.exp(np.cumsum(returns, axis=1))


def train(
    model, 
    seq_len,
    horizon,
    max_epoch, 
    batch_size,
    start_date="2010-01-01",
    end_date: str | None="2019-12-31",
    loss=nn.MSELoss(),
    ticker="PSEI.PS",
    optimizer=optim.Adam, 
    lr=1e-4,
    scale_method="std",
    device="cpu",
    nan: str = "drop"
):

    if end_date is None:
        end_date = now.strftime(f"%Y-%m-%d")

    _, X, y, scaling = prep_data.log_returns(
        seq_len=seq_len, 
        horizon=horizon,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        nan=nan
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
    
    df_train = {"y": train_test["y"]["Log Returns"]["train"].squeeze(2).squeeze(1).numpy()}
    df_test = {"y": train_test["y"]["Log Returns"]["test"].squeeze(2).squeeze(1).numpy()}
    
    if batch_size != 0:
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

    X_test = scaling.fit(train_test["X"]["Log Returns"]["test"]).to(device) # unnormed; needs to be normed; scale fit
    y_test = scaling.fit(train_test["y"]["Log Returns"]["test"]).to(device) # unnormed; needs to be normed; scale fit

    X_test_time = train_test["X"]["Timestamp"]["test"].to(device)
    y_test_time = train_test["y"]["Timestamp"]["test"].to(device)

    best_loss = float('inf')
    best_model = None
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"num_params: {num_params}")
    for epoch in range(max_epoch):
        start_time_wall, start_time_proc = time.time(), time.process_time()
        model.train()
        y_train_pred = torch.Tensor()  # Whole sequence of train dataset prediction
        
        if batch_size != 0:
            total_loss = 0.
            for X_batch, X_batch_time, y_batch, y_batch_time in loader:
                # Forward pass and backpropagation
                y_pred = model(  # normed result
                    src=scaling.fit(X_batch).to(device),  # unnormed; needs to be normed; scale fit
                    tgt=scaling.fit(y_batch).to(device),  # unnormed; needs to be normed; scale fit
                    src_time=X_batch_time.to(device), 
                    tgt_time=y_batch_time.to(device)
                )  # y_pred: (batch_size, horizon, 1)
                batch_loss = loss(y_pred, scaling.fit(y_batch).to(device))  # normed + unnormed, scale fit the unnormed
                total_loss += batch_loss
                backprop.zero_grad()
                batch_loss.backward()
                backprop.step()
                # Recording predictions
                y_train_pred = torch.cat(  # y_train pred unnormed, y_pred is still normed
                    (
                        y_train_pred.cpu(), # unnormed
                        scaling.inverse_fit(y_pred).squeeze(2).squeeze(1).cpu() # normed, needs scaler inverse fit
                    ), 
                    0
                )
                torch.cuda.empty_cache() 
        
        else:
            y_pred = model(
                src=scaling.fit(train_test["X"]["Log Returns"]["train"]).to(device), 
                tgt=scaling.fit(train_test["y"]["Log Returns"]["train"]).to(device), 
                src_time=scaling.fit(train_test["X"]["Timestamp"]["train"]).to(device), 
                tgt_time=scaling.fit(train_test["y"]["Timestamp"]["train"]).to(device),
            )
            total_loss = loss(
                y_pred, 
                scaling.fit(train_test["y"]["Log Returns"]["train"]).to(device)
            )
            backprop.zero_grad()
            total_loss.backward()
            backprop.step()
            y_train_pred = y_pred
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
            
            df_test[f"epoch_{epoch}"] = scaling.inverse_fit(y_test_pred).cpu().squeeze(2).squeeze(1).detach().numpy()

            test_loss = loss(y_test_pred, y_test)
            
            if best_loss < test_loss:
                best_loss = test_loss
                best_model = model

            torch.cuda.empty_cache() 

        train_loss, test_loss = total_loss.cpu().detach().numpy()/len(loader),  test_loss.cpu().detach().numpy()
        # Append train and test loss
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | test_loss {test_loss:.4f} | wall_time {wall_time:.4f} | process_time {proc_time:.4f}")

    # print(type(train_loss_list[1]), type(test_loss_list[1]))
    if horizon == 1:
        df_train = pd.DataFrame(df_train)
        df_test = pd.DataFrame(df_test)
        
    loss_df = pd.DataFrame(
        {
            "train_loss": train_loss_list,
            "test_loss": test_loss_list,
            "proc_time": proc_time_list,
            "wall_time": wall_time_list,  
        }
    )

    return model, best_model, loss_df, df_train, df_test, scaling

def test(
    model, 
    seq_len,
    horizon, 
    batch_size,
    start_date="2000-01-01",
    end_date="2022-12-31",
    loss=nn.MSELoss(),
    lr=1e-4,
    ticker="PSEI.PS",
    scale_method="std",
    device="cpu",
    nan="drop"
):

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    _, X, y, scaling = prep_data.log_returns(
        seq_len=seq_len, 
        horizon=horizon,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        nan=nan
    )    
    for X_key, y_key in zip(X, y):
        X[X_key] = prep_data.to_tensor(X[X_key])
        y[y_key] = prep_data.to_tensor(y[y_key])
    
    df_test = {"y": y["Log Returns"].cpu().squeeze(2).squeeze(1).detach().numpy()}

    if batch_size != 0:
        loader = DataLoader(
            list(zip(
                X["Log Returns"], 
                X["Timestamp"], 
                y["Log Returns"], 
                y["Timestamp"]
                )
            ), 
            batch_size=batch_size
        )
    model.to(device)
    with torch.no_grad():
        if batch_size == 0:
            y_pred = model(
                src=scaling.fit(X["Log Returns"]).to(device),
                src_time=X["Timestamp"].to(device),
                tgt=scaling.fit(y["Log Returns"]).to(device),
                tgt_time=y["Timestamp"].to(device)
            )
            torch.cuda.empty_cache()
        else:
            y_pred = torch.Tensor()
            for X_batch, X_batch_time, y_batch, y_batch_time in loader:
                # Forward pass and backpropagation
                y_batch_pred = model(  # normed result
                    src=scaling.fit(X_batch).to(device),  # unnormed; needs to be normed; scale fit
                    tgt=scaling.fit(y_batch).to(device),  # unnormed; needs to be normed; scale fit
                    src_time=X_batch_time.to(device), 
                    tgt_time=y_batch_time.to(device)
                )  # y_pred: (batch_size, horizon, 1)
                y_pred = torch.cat( 
                    (
                        y_pred.cpu(), # unnormed
                        y_batch_pred.cpu() # normed, needs scaler inverse fit
                    ), 
                    0
                )
            print(y_pred.shape)

    count_y_pred= scaling.inverse_fit(y_pred.cpu()).detach().numpy()
    count_y = y["Log Returns"].cpu().detach().numpy()

    test_loss = {
        "mse": float(mse(y_pred.to(device), scaling.fit(y["Log Returns"].to(device))).cpu().detach().numpy()),
        "mae": float(mae(y_pred.to(device), scaling.fit(y["Log Returns"].to(device))).cpu().detach().numpy()),
        "mape": float(torch.nanmean(
            torch.abs(
                (
                    y_pred.to(device)-scaling.fit(y["Log Returns"].to(device))
                )/(
                    scaling.fit(y["Log Returns"].to(device))
                )
                )
            ).cpu().detach().numpy()),
        "rmse": float(torch.sqrt(
            mse(y_pred.to(device), scaling.fit(y["Log Returns"].to(device)))
        ).cpu().detach().numpy()),
        "accuracy": {
            "bd": float(((count_y_pred * count_y) > 0).sum() / (count_y != 0).sum()),
            "ffill": float(((count_y_pred * count_y) >= 0).sum() / (np.size(count_y)))
        }
        
    }
    # loss(y_pred.to(device), scaling.fit(y["Log Returns"].to(device)))
    df_test["pred"] = scaling.inverse_fit(y_pred).cpu().squeeze(2).squeeze(1).detach().numpy()
    df_test

    return df_test, test_loss

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
