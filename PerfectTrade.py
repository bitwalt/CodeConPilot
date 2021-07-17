"""Download weekly crypto data """

import os
import sys
import datetime
import requests
import pandas as pd
import numpy as np

API_URL = "https://min-api.cryptocompare.com/data/histoday"

def download_data():
    """Download crypto data from Cryptocompare API"""
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": 2000,
        "aggregate": 1
    }

    df = pd.DataFrame()
    for _ in range(2000):
        response = requests.get(API_URL, params)
        if "Response" in response.json():
            break
        else:
            df = df.append(response.json()["Data"], ignore_index=True)
        params["toTs"] = response.json()["TimeFrom"]
    
    return df

def find_best_trade():
    """Find the best trade for a given crypto"""
    df = download_data()
    df = df[df["time"] > datetime.datetime.now() - datetime.timedelta(days=7)]
    df = df.set_index(["time"])
    df = df.sort_index()
    df = df.drop(["time", "volumefrom", "volumeto"], axis=1)
    df = df.resample("1D").last()

    # Find the best trade
    best_trade = df.iloc[0]
    best_trade["profit"] = best_trade["close"] - best_trade["open"]
    best_trade["profit"] = best_trade["profit"] / best_trade["open"]
    best_trade["profit"] = best_trade["profit"] * 100
    best_trade["profit"] = np.round(best_trade["profit"], 2)
    best_trade["profit"] = "%.2f" % best_trade["profit"] + "%"
    return best_trade



