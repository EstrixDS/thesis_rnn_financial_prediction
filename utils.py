# import necessary libraries
import datetime
import math
import pandas as pd
import yfinance as yf
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns


def download() -> pd.DataFrame:
    """Downloads data from yfinance

    Returns:
        pd.Dataframe: Dataframe with OHLC-Data including Adj. CLose and Volume of the german stock index from 1990 to March 31st 2022, processed with forward_fill
    """
    df = yf.download("^GDAXI", start="1990-01-01", end="2022-03-31", progress=False)
    df.index = pd.to_datetime(df.index)
    dates = pd.date_range(start="1990-01-01", end="2022-03-31")
    df = df.reindex(dates, fill_value=0)
    df.replace(0, np.nan, inplace=True)
    df = df.ffill(axis=0)
    return df


def process(df: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    """preprocesses the raw dataframe and adds the label-column 'volatility' to the dataframe

    Args:
        df (pd.DataFrame): the unprocessed Dataframe from Yahoo! Finance
        trading_days (int): Integer for the volatility calculation (30=1 month,20= 1 trading month, 365=1 year, 255= 1 trading year)

    Returns:
        df = pd.DataFrame: DataFrame with the original data and the two columns "returns" and "volatility"
        vol = ArrayLike: Array of all the volatility values
    """
    # compute daily returns and day moving historical volatility based on the trading_days
    df["returns"] = np.log(df["Close"] / df["Close"].shift(-1))
    df["volatility"] = df["returns"].rolling(window=trading_days).std() ** 0.5
    df = df.dropna()
    vol = df["volatility"].values
    return vol


def test(x: np.ndarray):
    """Tests a ndarray with the Augmented-Dickey-Fuller Test

    Args:
        x (np.ndarray): Numpy nd.array including the timeline
    """
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))


def convert2matrix(data_arr, look_back):
    """converts a matrix with an amount of look_back

    Args:
        data_arr (np.ndarray): Data Array including the Volatility Data
        look_back (int): Amount of Days to look back

    Returns:
        np.array: Numpy Array with a Matrix, where each point is a list of volatilty values used for training
        np.array: Numpy Array with volatility values used as labels
    """
    X, Y = [], []
    for i in range(len(data_arr) - look_back):
        d = i + look_back
        X.append(data_arr[i:d])
        Y.append(data_arr[d])
    return np.array(X), np.array(Y)


def compute_prediction_frame(y_labels: list, predicted: list):
    """creates a DataFrame with the two Columns of the Actual Volatility Data and the Predicted Volatililty Values

    Args:
        y_labels (list): List of Actual Volatility Values
        predicted (list): List of Predicted Volatility Values

    Returns:
        pd.DataFrame: DataFrame with the Actual and the Predicted Volatility Values
    """
    df = pd.DataFrame(data=[y_labels, predicted], dtype=float).transpose()
    df.columns = ["Actual", "Prediction"]
    return df


def print_error(y_train, y_test, train_predict, test_predict):
    """Calculates and prints the Training and Testing Root Mean Squared Error

    Args:
        y_train (List): Labels of the Training_dataset
        y_test (List): Labels of the test_dataset
        train_predict (List): List of the Predicted Volatility Values based on the training values
        test_predict (List): List of the Predicted Volatilty Values based on the testing values
    """
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
    # Print RMSE
    print("Train RMSE: %.3f RMSE" % (train_rmse))
    print("Test RMSE: %.3f RMSE" % (test_rmse))


# Compute the Pearson correlation
def compute_Pearson_correlation(df, content_type, do_print):
    """computes the pearson correlation of the Dataframe

    Args:
        df (pd.DataFrame): DataFrame with the Actual and the Predicted Volatility Values
        content_type (str): String of the Content Type
        do_print (Boolean): If True, it prints the pearson correlation, if False, it does not

    Returns:
        int: Value of the result of the Pearson Correlation
    """

    corr_value = df.corr(method="pearson")
    if do_print:
        print("---------------------------------\n")
        print("Pearson correlation for " + content_type + " data set\n", corr_value)
        print("---------------------------------\n")

    return corr_value["Prediction"][0]


# plot as scatter plot and compute a regression line, including alpha and beta
def scatterplotRegression(df, model_type, EPOCHS, BATCH_SIZE):
    """Plots a Regressionplot with a text annotation containing additional information

    Args:
        df (pd.DataFrame): DataFrame consisting of the Actual and the Predicted Volatility Values
        model_type (str): Name of the model
        EPOCHS (int): the amount of epochs, the model was trained on
        BATCH_SIZE (int): number of training examples

        It does not return anything, but plots a regression plot and saves it
    """
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


def error_evaluation(epochs, loss, val_loss, model_type):
    """Plots the development of the training and validation error

    Args:
        epochs (int): Amount of Epochs used in training
        loss (list): list of the training loss values per epoch
        val_loss (list): list of the validation loss values per epoch
        model_type (str): Name of the Model

        It does not return anything, but plots the Training and Validation Loss of the model
    """
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss:", model_type)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss Score")
    plt.show()
