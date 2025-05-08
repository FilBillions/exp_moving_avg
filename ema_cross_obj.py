import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta


class ExpMovingAverageTable():
        def __init__(self, ma1, ma2, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d"):
                df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
                self.df = df
                self.ticker = ticker
                self.ma1 = ma1
                self.ma2 = ma2

        def run_algo(self, print_table=True):
                #adding day count
                day_count = np.arange(1, len(self.df) + 1)
                self.df['Day Count'] = day_count
                #dropping unnecessary columns
                if 'Volume' in self.df.columns:
                        self.df.drop(columns=['Volume'], inplace = True)
                if 'Capital Gains' in self.df.columns:
                        self.df.drop(columns=['Capital Gains'], inplace = True)
                if 'Dividends' in self.df.columns:
                        self.df.drop(columns=['Dividends'], inplace = True)
                if 'Stock Splits' in self.df.columns:
                        self.df.drop(columns=['Stock Splits'], inplace = True)

                # --- INITIALISE THE DATAFRAME ---
                # ---
                self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
                self.df['Cumulative Return %'] = (np.exp(self.df['Return %'] / 100).cumprod() - 1) * 100

                # ---
                # initialise EMA1 and EMA2
                self.df[f'{self.ma1}-day EMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
                self.df[f'{self.ma2}-day EMA'] = self.df['Close'].rolling(int(self.ma2)).mean().shift()

                # Calculate EMA for the rest of the rows starting from the `ma1`th day
                for i in range(int(self.ma1) + 1, len(self.df)):
                        self.df.iloc[i, self.df.columns.get_loc(f'{self.ma1}-day EMA')] = (
                        (self.df.iloc[i - 1, self.df.columns.get_loc('Close')] * (2 / (int(self.ma1) + 1))) +
                        (self.df.iloc[i - 1, self.df.columns.get_loc(f'{self.ma1}-day EMA')] * (1 - (2 / (int(self.ma1) + 1))))
                        )

                for i in range(int(self.ma2) + 1, len(self.df)):
                        self.df.iloc[i, self.df.columns.get_loc(f'{self.ma2}-day EMA')] = (
                        (self.df.iloc[i - 1, self.df.columns.get_loc('Close')] * (2 / (int(self.ma2) + 1))) +
                        (self.df.iloc[i - 1, self.df.columns.get_loc(f'{self.ma2}-day EMA')] * (1 - (2 / (int(self.ma2) + 1))))
                        )

                # Signal to long
                self.df['Signal'] = np.where(self.df[f'{self.ma1}-day EMA'] > self.df[f'{self.ma2}-day EMA'], 1, 0)

                # Signal to short
                self.df['Signal'] = np.where(self.df[f'{self.ma1}-day EMA'] < self.df[f'{self.ma2}-day EMA'], -1, self.df['Signal'])

                # Model Return %
                self.df['EMA Model Return %'] = self.df['Return %'] * self.df['Signal']

                # Entry column for visualization
                self.df['Entry'] = self.df.Signal.diff()

                # drop rows
                self.df.dropna(inplace=True)

                # Cumulative Returns
                self.df['Cumulative EMA Model Return %'] = (np.exp(self.df['EMA Model Return %'] / 100).cumprod() - 1) * 100

                # Recalculate Return % and cumulative Return % to include model Return %s
                self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
                self.df['Cumulative Return %'] = (np.exp(self.df['Return %'] / 100).cumprod() - 1) * 100

                # Formatting the table
                self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', f'{self.ma1}-day EMA', f'{self.ma2}-day EMA', 'Return %', 'Cumulative Return %', 'EMA Model Return %', 'Cumulative EMA Model Return %', 'Signal', 'Entry']]), 3)
                if print_table:
                        #options to show all rows and columns
                        #pd.set_option('display.max_rows', None)
                        #pd.set_option('display.max_columns', None)
                        #pd.set_option('display.width', None)
                        #pd.set_option('display.max_colwidth', None)
                        return self.df
                pass

        def gen_visual(self):
                model_days = self.df['Day Count'].iloc[-1] - self.df['Day Count'].iloc[0] + 1
                #parameters for grid size
                self.df.index = pd.to_datetime(self.df.index).strftime('%Y-%m-%d-%H:%M')
                fig = plt.figure(figsize=(12, 8))

                # Use the actual index for x-values
                x_values = range(len(self.df.iloc[-model_days:]))

                #plot ticker closing prices and MAs, .iloc for integers
                plt.plot(x_values, self.df.iloc[-model_days:]['Close'])
                plt.plot(x_values, self.df.iloc[-model_days:][f'{self.ma1}-day EMA'], label = f'{self.ma1}-day EMA')
                plt.plot(x_values, self.df.iloc[-model_days:][f'{self.ma2}-day EMA'], label = f'{self.ma2}-day EMA')

                # Plot buy signals (Entry == 2)
                plt.scatter(
                        [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == 2],
                        self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Entry'] == 2],
                        marker='^', color='g', s=100, label='Buy Signal'
                )

                # Plot sell signals (Entry == -2)
                plt.scatter(
                        [x_values[i] for i in range(len(self.df.iloc[-model_days:])) if self.df.iloc[-model_days:].iloc[i]['Entry'] == -2],
                        self.df.iloc[-model_days:]['Close'][self.df.iloc[-model_days:]['Entry'] == -2],
                        marker='v', color='r', s=100, label='Sell Signal'
                )
                # Set x-axis to date values and make it so they dont spawn too many labels
                plt.xticks(ticks=x_values, labels=self.df.iloc[-model_days:].index, rotation=45)
                plt.locator_params(axis='x', nbins=10)

                # create grid
                plt.grid(True, alpha = .5)
                # plot legend
                plt.legend(['Close', f'{self.ma1}-day EMA', f'{self.ma2}-day EMA', 'Buy Signal', 'Sell Signal'], loc = 2)

                #print statements        
                print(f'from {self.df.index[-model_days]} to {self.df.index[-1]}')
                print(f'count of buy signals: {len(self.df[self.df["Entry"] == 2])}')
                print(f'count of sell signals: {len(self.df[self.df["Entry"] == -2])}')

        def gen_comp(self):
                labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
                fig1= plt.figure(figsize=(12, 6))
                x_values = range(len(self.df))

                # add buy/hold to legend if it doesn't exist
                if f'{self.ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
                        plt.plot(x_values, self.df['Cumulative Return %'], label=f'{self.ticker} Buy/Hold')
                # model plot
                plt.plot(x_values, self.df['Cumulative EMA Model Return %'], label=f'{self.ticker} EMA Model')

                # Set x-axis to date values and make it so they dont spawn too many labels
                plt.xticks(ticks=x_values, labels=labels, rotation=45)
                plt.locator_params(axis='x', nbins=10)

                # grid and legend
                plt.legend(loc=2)
                plt.grid(True, alpha=.5)
                # print cumulative Return % if not already printed
                print(f"{self.ticker} Cumulative EMA Model Return %:", round(self.df['Cumulative EMA Model Return %'].iloc[-1], 2))
                print(f" from {self.df.index[0]} to {self.df.index[-1]}")

        def sharpe(self):
                buyhold_avg_r = float((np.mean(self.df['Return %'])))
                buyhold_std = float((np.std(self.df['Return %'])))
                buyhold_sharpe = (buyhold_avg_r / buyhold_std) * 252 ** 0.5
                model_avg_r = float((np.mean(self.df['EMA Model Return %'])))
                model_std = float((np.std(self.df['EMA Model Return %'])))
                model_sharpe = (model_avg_r / model_std) * 252 ** 0.5
                print(f" Buy/Hold Sharpe {round(buyhold_sharpe, 3)}")
                print(f" Model Sharpe {round(model_sharpe, 3)}")