# yauber-backtester
Yet Another Universal Backtesting Engine Release (YAUBER) - Backtester

## Description
yauber-backtester is bare-bone portfolio backtesting engine that:
- supports various portfolio management techniques: asset ranking, basket trading, portfolio rebalancing, etc.
- intended to work on large asset universes (like 2000-3000 US Stocks EOD), or small intraday asset universes (like futures or forex, 1h timeframe). 
- supports meta-strategies, building and managing a portfolio of other trading strategies
- allows simulating capital allocations, costs, margin trading, etc.

## Limitations
1. This backtester is designed to trade only at regular prices, like close, next bar open, or next vwap price. 
This is a fundamental design principle, that allows fancy portfolio management logic, and simplifies trading strategy code.
2. This backtester will remain bare-bone core without any advanced features like optimization, advanced reporting, data feeds and order management part.
I don't have plans to open-source these modules. However, you can hire me and I can implement a customized version that fits your needs.

## Examples and documentation
Please follow the documentation strings and comments in the code and notebooks.

You can use Jupyter notebooks to develop and backtest trading strategies, here some examples:

- [Simple strategy (10 assets x 10 years x 1hr intraday)](https://github.com/alexveden/yauber-backtester/blob/master/notebooks/Strategy%2010%20assets%20intraday.ipynb)
- [Simple strategy (1000 assets x 10 years x Daily)](https://github.com/alexveden/yauber-backtester/blob/master/notebooks/Strategy%201000%20assets%20EOD.ipynb)
- [Meta-strategy of two Long and Short simple strategies](https://github.com/alexveden/yauber-backtester/blob/master/notebooks/Strategy%20composite.ipynb)

## Author
Aleksandr Vedeneev 2018 i@alexveden.com

MIT Licence