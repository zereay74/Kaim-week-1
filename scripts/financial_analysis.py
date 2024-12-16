import talib
import numpy as np
import pandas as pd
import pynance as pn
import matplotlib.pyplot as plt


class FinancialAnalysis:
    """
    A class to perform financial analysis using technical indicators with TA-Lib.
    """

    def __init__(self, data):
        """
        Initializes the FinancialAnalysis class with financial data.

        :param data: pd.DataFrame, DataFrame containing financial data with required columns.
        """
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        self.data = data

    def calculate_moving_average(self, period=14):
        """
        Calculates the moving average for the 'Close' price.

        :param period: int, the period for the moving average
        :return: pd.Series, moving average values
        """
        self.data[f'MA_{period}'] = talib.SMA(self.data['Close'], timeperiod=period)
        return self.data[f'MA_{period}']

    def calculate_rsi(self, period=14):
        """
        Calculates the Relative Strength Index (RSI) for the 'Close' price.

        :param period: int, the period for RSI calculation
        :return: pd.Series, RSI values
        """
        self.data[f'RSI_{period}'] = talib.RSI(self.data['Close'], timeperiod=period)
        return self.data[f'RSI_{period}']

    def calculate_macd(self):
        """
        Calculates the Moving Average Convergence Divergence (MACD) for the 'Close' price.

        :return: tuple (pd.Series, pd.Series, pd.Series), MACD line, signal line, and histogram
        """
        macd, macd_signal, macd_hist = talib.MACD(
            self.data['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        self.data['MACD_Hist'] = macd_hist
        return macd, macd_signal, macd_hist

# Example usage:
# df = pd.read_csv("path_to_your_data.csv")
# analysis = FinancialAnalysis(df)
# df['MA_20'] = analysis.calculate_moving_average(period=20)
# df['RSI_14'] = analysis.calculate_rsi(period=14)
# macd, macd_signal, macd_hist = analysis.calculate_macd()



class FinancialMetrics:
    """
    A class to calculate financial metrics using TA-Lib.
    """

    def __init__(self, data):
        """
        Initialize with financial data.

        :param data: pd.DataFrame, DataFrame containing at least a 'Close' column.
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")
        self.data = data

    def calculate_daily_returns(self):
        """
        Calculate daily returns based on the 'Close' price using TA-Lib.

        :return: pd.Series, daily returns
        """
        self.data['Daily_Returns'] = talib.ROCP(self.data['Close'], timeperiod=1)
        return self.data['Daily_Returns']

    def calculate_cumulative_returns(self):
        """
        Calculate cumulative returns based on daily returns.

        :return: pd.Series, cumulative returns
        """
        if 'Daily_Returns' not in self.data.columns:
            self.calculate_daily_returns()
        self.data['Cumulative_Returns'] = (1 + self.data['Daily_Returns']).cumprod() - 1
        return self.data['Cumulative_Returns']

    def calculate_volatility(self, period=252):
        """
        Calculate the annualized volatility of daily returns using TA-Lib.

        :param period: int, the number of trading days in a year (default is 252)
        :return: float, annualized volatility
        """
        if 'Daily_Returns' not in self.data.columns:
            self.calculate_daily_returns()

        # Calculate standard deviation
        volatility = talib.STDDEV(self.data['Daily_Returns'], timeperiod=period) * np.sqrt(period)
        
        # Handle cases where the last value might be NaN
        valid_volatility = volatility[~np.isnan(volatility)]
        return valid_volatility.iloc[-1] if not valid_volatility.empty else np.nan


    def calculate_sharpe_ratio(self, risk_free_rate=0.01, period=252):
        """
        Calculate the Sharpe ratio, a measure of risk-adjusted return using TA-Lib.

        :param risk_free_rate: float, risk-free rate of return (default is 1%)
        :param period: int, the number of trading days in a year (default is 252)
        :return: float, Sharpe ratio
        """
        if 'Daily_Returns' not in self.data.columns:
            self.calculate_daily_returns()

        # Calculate average daily return using SMA
        sma = talib.SMA(self.data['Daily_Returns'], timeperiod=period)
        valid_sma = sma[~np.isnan(sma)]
        
        if valid_sma.empty:
            return np.nan  # Return NaN if no valid SMA exists

        # Calculate excess return
        avg_daily_return = valid_sma.iloc[-1]
        excess_return = avg_daily_return * period - risk_free_rate

        # Calculate volatility
        volatility = self.calculate_volatility(period=period)

        # Avoid division by zero
        sharpe_ratio = excess_return / volatility if volatility != 0 else np.nan
        return sharpe_ratio

# Example usage:
# df = pd.read_csv("path_to_your_data.csv")
# metrics = FinancialMetrics(df)
# df['Daily_Returns'] = metrics.calculate_daily_returns()
# df['Cumulative_Returns'] = metrics.calculate_cumulative_returns()
# volatility = metrics.calculate_volatility()
# sharpe_ratio = metrics.calculate_sharpe_ratio(risk_free_rate=0.03)





class FinancialVisualizer:
    """
    A class to visualize financial data and technical indicators.
    """

    def __init__(self, data):
        """
        Initialize with financial data.

        :param data: pd.DataFrame, DataFrame containing the financial and indicator data.
        """
        self.data = data

    def plot(self, columns, title, xlabel="Date", ylabel="Value", figsize=(12, 6), kind="line"):
        """
        General plot function for reducing repetitive plotting code.

        :param columns: list, columns to plot.
        :param title: str, title of the plot.
        :param xlabel: str, label for the x-axis.
        :param ylabel: str, label for the y-axis.
        :param figsize: tuple, size of the plot.
        :param kind: str, type of plot (e.g., 'line', 'bar', 'scatter').
        """
        plt.figure(figsize=figsize)
        if kind == "line":
            for column in columns:
                plt.plot(self.data['Date'], self.data[column], label=column)
        elif kind == "bar":
            for column in columns:
                plt.bar(self.data['Date'], self.data[column], label=column)
        elif kind == "scatter":
            for column in columns:
                plt.scatter(self.data['Date'], self.data[column], label=column)
        else:
            raise ValueError(f"Unsupported plot kind: {kind}")
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_price_and_moving_average(self):
        """
        Plot the stock price and the moving average.
        """
        self.plot(columns=['Close', 'MA_20'], 
                  title="Stock Price and 20-Day Moving Average",
                  ylabel="Price")

    def plot_rsi(self):
        """
        Plot the Relative Strength Index (RSI).
        """
        self.plot(columns=['RSI_14'], 
                  title="Relative Strength Index (RSI)",
                  ylabel="RSI Value")

    def plot_macd(self):
        """
        Plot the MACD and its signal and histogram.
        """
        self.plot(columns=['MACD', 'MACD_Signal', 'MACD_Hist'], 
                  title="MACD, Signal Line, and Histogram",
                  ylabel="MACD Value")

    def plot_returns(self):
        """
        Plot daily and cumulative returns.
        """
        self.plot(columns=['Daily_Returns', 'Cumulative_Returns'], 
                  title="Daily and Cumulative Returns",
                  ylabel="Return")

    def plot_custom(self, columns, kind="line", title="Custom Plot", xlabel="Date", ylabel="Value", figsize=(12, 6)):
        """
        Plot user-specified columns with a chosen plot type.

        :param columns: list, columns to plot.
        :param kind: str, type of plot (e.g., 'line', 'bar', 'scatter').
        :param title: str, title of the plot.
        :param xlabel: str, label for the x-axis.
        :param ylabel: str, label for the y-axis.
        :param figsize: tuple, size of the plot.
        """
        self.plot(columns=columns, title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize, kind=kind)

# Example usage:
# visualizer = FinancialVisualizer(data)
# visualizer.plot_price_and_moving_average()
# visualizer.plot_rsi()
# visualizer.plot_macd()
# visualizer.plot_returns()
# visualizer.plot_custom(columns=['Volume'], kind='bar', title='Trading Volume')
