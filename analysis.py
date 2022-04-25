import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")


def data_download():
    """_summary_

    Returns:
        _type_: _description_
    """
    price_data_dax = yf.download("^GDAXI", start="2019-1-1", auto_adjust=True)
    price_data_dax.index = pd.to_datetime(price_data_dax.index)
    # compute daily returns and 20 day moving historical volatility
    price_data_dax["deviation"] = price_data_dax["High"] - price_data_dax["Low"]
    price_data_dax["volatility"] = (
        price_data_dax["deviation"] - price_data_dax["deviation"].std()
    )
    return price_data_dax


def plot_raw(price_data_dax: pd.DataFrame):
    """_summary_

    Args:
        price_data_dax (_type_): _description_
    """
    # Plot the close price
    plt.figure(figsize=(10, 7))
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.hist(price_data_dax["volatility"])
    plt.title("Distribution of Volatility")
    plt.subplot(122)
    plt.hist(np.log(price_data_dax["volatility"]))
    plt.title("Log Transformation of Volatility")
    plt.show()


def stabilitycheck(X):
    """_summary_

    Args:
        X (_type_): _description_
    """
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(X)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


def main():
    """_summary_"""
    price_data_dax = data_download()
    X = price_data_dax["volatility"].values
    stabilitycheck(X)
    return price_data_dax


if __name__ == "__main__":
    main()
