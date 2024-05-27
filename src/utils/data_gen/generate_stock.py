from yahoofinancials import YahooFinancials
import pandas as pd
import numpy as np
import tqdm
import argparse
import os

def process_stock_price(tickers, start_date, end_date):
    yfs = [YahooFinancials(t) for t in tickers]
    X = []
    dates = []
    for yf in tqdm.tqdm(yfs, "Processing stock market data"):
        r = yf.get_historical_price_data(start_date, end_date, 'daily')
        r = dict(r)
        ticker = list(r.keys())[0]
        close_prices = np.array([t['close'] for t in r[ticker]['prices']])
        X.append(close_prices)
        dates.append([t['formatted_date'] for t in r[ticker]['prices']])
    X = np.array(X)
    # print(X.shape)
    dates = np.array(dates)
    return X, dates


def get_log_returns(X):
    return np.diff(np.log(X))

def standardize(X):
    mu = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    mu = np.repeat(mu[:, np.newaxis], repeats=X.shape[1], axis=1)
    std = np.repeat(std[:, np.newaxis], repeats=X.shape[1], axis=1)
    return (X-mu)/std

def generate_stock(args):
    df = pd.read_csv(args.stock_list_file)
    tickers = df['Symbol'].tolist()

    X, dates = process_stock_price(tickers, args.start_date, args.end_date)
    X = get_log_returns(X)

    X = standardize(X)
    
    D = X.shape[0]
    T = X.shape[1]
    L = args.chunk_size
    dat = np.zeros((D, T//L, L))
    date_array = []
    
    for i in range(T//L):
        dat[:, i] = X[:, L*i:(i+1)*L]
        date_array.append(dates[0][L*i])
        print(date_array[-1])
    date_array.append(args.end_date)
    print("Data shape:", dat.shape)
    dat = dat.transpose((1, 2, 0))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    np.save(os.path.join(args.save_dir, 'X.npy'), dat)

    for sector in df['Sector'].unique():
        X_sector = dat[:, :, df[df['Sector'] == sector].index.values]
        np.save(os.path.join(args.save_dir, f'X_{sector.replace(" ", "")}.npy'), X_sector)

    with open(os.path.join(args.save_dir, 'dates.csv'), 'w') as f:
        for i in range(len(date_array)-1):
            f.write(f"{date_array[i]} to {date_array[i+1]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Stock data generator from Yahoo Financials")
    parser.add_argument("--start_date", type=str, default='2016-01-01')
    parser.add_argument("--end_date", type=str, default='2023-07-01')
    parser.add_argument('--chunk_size', type=int, default=31, help='Number of days to chunk together into one sample. Default is 31 days')
    parser.add_argument('--stock_list_file', type=str)
    parser.add_argument('--save_dir', type=str)
    
    args = parser.parse_args()
    generate_stock(args)
