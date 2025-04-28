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
                return self.df
            pass

        def gen_ema_cross_visual(self, model_days,):
        #parameters for grid size
            plt.rcParams['figure.figsize'] = 12, 8
        #create grid
            plt.grid(True, alpha = .5)
        #plot ticker closing prices and MAs, .iloc for integers
            plt.plot(self.df.iloc[-model_days:]['Close'])
            plt.plot(self.df.iloc[-model_days:][f'{self.ma1}-day EMA'], label = f'{self.ma1}-day EMA')
            plt.plot(self.df.iloc[-model_days:][f'{self.ma2}-day EMA'], label = f'{self.ma2}-day EMA')
        #plotting entry points, .loc for labels
            plt.plot(self.df[-model_days:].loc[self.df.Entry == 2].index, self.df[-model_days:][f'{self.ma1}-day EMA'][self.df.Entry == 2], '^', color = 'g', markersize = 10)
            plt.plot(self.df[-model_days:].loc[self.df.Entry == -2].index, self.df[-model_days:][f'{self.ma2}-day EMA'][self.df.Entry == -2], 'v', color = 'r', markersize = 10)
        #plot legend
            plt.legend(['Close', f'{self.ma1}-day EMA', f'{self.ma2}-day EMA', 'Buy Signal', 'Sell Signal'], loc = 2)
       
        def gen_buyhold_comp(self, ticker):
        #add buy/hold to legend if it doesn't exist
            if f'{ticker} Buy/Hold' not in [line.get_label() for line in plt.gca().get_lines()]:
                plt.plot(self.df['Cumulative Return'], label=f'{ticker} Buy/Hold')
        #model plot
            plt.plot(self.df['Cumulative EMA Model Return'], label=f'{ticker} EMA Model')
            plt.legend(loc=2)
            plt.grid(True, alpha=.5)
        #print cumulative return if not already printed
            print(f"{ticker} Cumulative EMA Model Return:", round(self.df['Cumulative EMA Model Return'].iloc[-1], 2))