import pandas as pd
import yfinance as yf
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


def download():
    df = yf.download("^GDAXI", start="1990-01-01", end="2022-03-31", progress=False)
    df.index = pd.to_datetime(df.index)
    dates = pd.date_range(start="1990-01-01", end="2022-03-31")
    df = df.reindex(dates, fill_value=0)
    df.replace(0, np.nan, inplace=True)
    df = df.ffill(axis=0)
    return df


def process(df: pd.DataFrame, trading_days: int):
    # compute daily returns and 20 day moving historical volatility
    df["returns"] = np.log(df["Close"] / df["Close"].shift())
    df["volatility"] = df["returns"].rolling(window=20).std() * np.sqrt(trading_days)
    df = df.dropna()
    X = df["volatility"].values
    df.to_csv("Values_new.csv")
    return df, X


def test(x: np.ndarray):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


def convert2matrix(data_arr, look_back):
    X, Y = [], []
    for i in range(len(data_arr) - look_back):
        d = i + look_back
        X.append(data_arr[i:d])
        Y.append(data_arr[d])
    return np.array(X), np.array(Y)


def compute_prediction_frame(y_labels: list, predicted: list):
    df = pd.DataFrame(data=[y_labels, predicted], dtype=float).transpose()
    df.columns = ["Actual", "Prediction"]
    return df


# Compute the Pearson correlation
def compute_Pearson_correlation(df, content_type, do_print):

    corr_value = df.corr(method="pearson")
    if do_print:
        print("---------------------------------\n")
        print("Pearson correlation for " + content_type + " data set\n", corr_value)
        print("---------------------------------\n")

    return corr_value["Prediction"][0]


# plot as scatter plot and compute a regression line, including alpha and beta
def scatterplotRegression(df, model_type, EPOCHS, BATCH_SIZE):

    beta, alpha = np.polyfit(df.Actual, df.Prediction, 1)
    pearson_corr = compute_Pearson_correlation(
        df=df, content_type="TEST", do_print=False
    )
    parameter_text = "\n".join(
        (
            r"$epochs=%.0f$" % (EPOCHS,),
            r"$batch \ size=%.0f$" % (BATCH_SIZE,),
            r"$pearson\ corr:r}=%.2f$" % (pearson_corr,),
            r"$slope: \beta=%.2f$" % (beta,),
            r"$intercept: \alpha=%.2f$" % (alpha,),
        )
    )
    sns.set(rc={"figure.figsize": (15, 10)})
    sns.regplot(
        data=df,
        x="Actual",
        y="Prediction",
        x_ci="sd",
        scatter=True,
        line_kws={"color": "red"},
    )
    # add text annotation
    # set the textbox color to purple
    purple_rgb = (255.0 / 255.0, 2.0 / 255.0, 255.0 / 255.0)

    plt.annotate(
        parameter_text,
        xy=(0.05, 0.75),
        xycoords="axes fraction",
        ha="left",
        fontsize=14,
        color=purple_rgb,
        backgroundcolor="w",
    )
    plt.grid(True)
    # finally save the chart to disk
    plt.savefig(
        "./images/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "_Scatterplot_"
        + model_type
        + "_one-sd"
        + ".png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    # print a few regression statistics
    print("Beta of ", df.Prediction.name, " =", round(beta, 4))
    print("Alpha of ", df.Prediction.name, " =", round(alpha, 4))


def error_evaluation(epochs, mse, val_mse, loss, val_loss):
    plt.plot(epochs, mse, "bo", label="Training mse")
    plt.plot(epochs, val_mse, "b", label="Validation mse")
    plt.title("Training and validation Mean Squared Error")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()
