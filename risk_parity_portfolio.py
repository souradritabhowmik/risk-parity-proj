import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class RiskParityPortfolio:
    def __init__(self, tickers, start_date, end_date, initial_capital=10000):
        """
        Initialize Risk Parity Portfolio.
        :param tickers: List of stock tickers or ETFs
        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param initial_capital: starting capital
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.weights = None
        self.portfolio = None

    def fetch_data(self):
        """Download historical adjusted close prices."""
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        self.data.dropna(inplace=True)

    def calculate_volatility(self):
        """Calculate annualized volatility of returns for each asset."""
        returns = self.data.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        return vol

    def calculate_risk_parity_weights(self):
        """Calculate weights inversely proportional to volatility."""
        vol = self.calculate_volatility()
        inv_vol = 1 / vol
        self.weights = inv_vol / np.sum(inv_vol)
        return self.weights

    def backtest_portfolio(self):
        """Backtest the portfolio and compute cumulative returns."""
        returns = self.data.pct_change().dropna()
        weights_array = np.array(self.weights)
        portfolio_returns = returns.dot(weights_array)
        self.portfolio = pd.DataFrame()
        self.portfolio['Returns'] = portfolio_returns
        self.portfolio['Cumulative'] = (1 + portfolio_returns).cumprod() * self.initial_capital

    def plot_portfolio(self):
        """Plot cumulative portfolio value and individual asset contributions."""
        plt.figure(figsize=(14,7))
        plt.plot(self.portfolio['Cumulative'], label='Risk Parity Portfolio', color='purple')
        plt.title('Risk Parity Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def performance_metrics(self):
        """Calculate basic performance metrics."""
        total_return = self.portfolio['Cumulative'].iloc[-1] / self.initial_capital - 1
        annualized_return = (1 + total_return) ** (252 / len(self.portfolio)) - 1
        annualized_vol = self.portfolio['Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol
        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Annualized Volatility": annualized_vol,
            "Sharpe Ratio": sharpe_ratio
        }

if __name__ == "__main__":
    tickers = ['SPY', 'AGG', 'GLD']
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    rp = RiskParityPortfolio(tickers, start_date, end_date)
    rp.fetch_data()
    weights = rp.calculate_risk_parity_weights()
    print("Risk Parity Weights:")
    for t, w in zip(tickers, weights):
        print(f"{t}: {w:.2%}")
    rp.backtest_portfolio()
    rp.plot_portfolio()
    metrics = rp.performance_metrics()
    print("\nPortfolio Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2%}")
