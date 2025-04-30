import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fin_table_obj import Table

class ExpMovingAverageTable(Table):
    def __init__(self, df, ma1, ma2):
        super().__init__(df)
        self.ma1 = ma1
        self.ma2 = ma2

    def gen_table(self, optional_bool=True):
        super().gen_table()

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

        # Model return
        self.df['EMA Model Return'] = self.df['Return'] * self.df['Signal']

        # Entry column for visualization
        self.df['Entry'] = self.df.Signal.diff()

        # drop rows
        self.df.dropna(inplace=True)

        # Cumulative Returns
        self.df['Cumulative EMA Model Return'] = (np.exp(self.df['EMA Model Return'] / 100).cumprod() - 1) * 100

        # Recalculate return and cumulative return to include model returns
        self.df['Return'] = (np.log(self.df['Close']).diff()) * 100
        self.df['Cumulative Return'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100

        # Formatting the table
        self.df = round((self.df[['Day Count', 'Open', 'High', 'Low', 'Close', f'{self.ma1}-day EMA', f'{self.ma2}-day EMA', 'Return', 'Cumulative Return', 'EMA Model Return', 'Cumulative EMA Model Return', 'Signal', 'Entry']]), 3)
        if optional_bool:
            #options to show all rows and columns
            #pd.set_option('display.max_rows', None)
            #pd.set_option('display.max_columns', None)
            #pd.set_option('display.width', None)
            #pd.set_option('display.max_colwidth', None)
            return self.df
        pass

    def gen_ema_cross_visual(self, model_days,):
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
        print(f'count of buy signals: {len(self.df[self.df["Entry"] == 2]) / 2}')
        print(f'count of sell signals: {len(self.df[self.df["Entry"] == -2]) / 2}')

    def gen_buyhold_comp(self, ticker):
        labels = pd.to_datetime(self.df.index).strftime('%Y-%m-%d')
        fig1= plt.figure(figsize=(12, 6))
        x_values = range(len(self.df))

# add buy/hold to legend if it doesn't exist
        if f'{ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
            plt.plot(x_values, self.df['Cumulative Return'], label=f'{ticker} Buy/Hold')
# model plot
        plt.plot(x_values, self.df['Cumulative EMA Model Return'], label=f'{ticker} MACD Model')

# Set x-axis to date values and make it so they dont spawn too many labels
        plt.xticks(ticks=x_values, labels=labels, rotation=45)
        plt.locator_params(axis='x', nbins=10)

# grid and legend
        plt.legend(loc=2)
        plt.grid(True, alpha=.5)
# print cumulative return if not already printed
        print(f"{ticker} Cumulative EMA Model Return:", round(self.df['Cumulative EMA Model Return'].iloc[-1], 2))
        print(f" from {self.df.index[0]} to {self.df.index[-1]}")