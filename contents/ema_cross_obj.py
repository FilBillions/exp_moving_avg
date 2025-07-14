import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from scipy import stats

np.set_printoptions(legacy='1.25')

class ExpMovingAverageTable():
    def __init__(self, ticker, ma1=50, ma2=250, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d", optional_df = None):
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(ticker, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
        self.ticker = ticker
        self.interval = interval
        self.ma1 = ma1
        self.ma2 = ma2
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count

        self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
        # ---
        # initialise EMA1 and EMA2
        self.df[f'{self.ma1}-day EMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
        self.df[f'{self.ma2}-day EMA'] = self.df['Close'].rolling(int(self.ma2)).mean().shift()
        self.df['Previous Close'] = self.df['Close'].shift(1)

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

    def run_algo(self, start_date=date.today().year- 1, end_date=date.today(), return_table=False):    
        if self.df.index.tz is not None:
            if pd.Timestamp(start_date).tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date).tz_convert('UTC')
            if pd.Timestamp(end_date).tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date).tz_convert('UTC')
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)
        if isinstance(start_date, int):
            post_data = self.df[self.df.index >= start_date]
            data_cutoff = []
        else:
            post_data = self.df[self.df.index >= start_date]
            if end_date == date.today():
                end_date = str(date.today())
            data_cutoff = self.df[self.df.index >= end_date]
        actions = []
        dates = []

        post_idx = self.df.index.get_loc(post_data.index[0])

        prev_action = None
        next_action = None  # This will hold the action for the next day
        action = 'No Action'  # <-- Ensure action is always initialized
        while post_idx < (len(self.df) - len(data_cutoff)):
            start_date = self.df.index[post_idx]
            if self.df[f'{self.ma1}-day EMA'].iloc[post_idx] > self.df[f'{self.ma2}-day EMA'].iloc[post_idx]:
                if prev_action == "Hold (Long)" or prev_action == "Buy":
                    action = "Hold (Long)"
                elif prev_action == "No Action" or prev_action == "Sell" or prev_action == "Hold (Short)":
                    action = "Buy"
            elif self.df[f'{self.ma1}-day EMA'].iloc[post_idx] < self.df[f'{self.ma2}-day EMA'].iloc[post_idx]:
                if prev_action == "Hold (Short)" or prev_action == "Sell":
                    action = "Hold (Short)"
                elif prev_action == "No Action" or prev_action == "Buy" or prev_action == "Hold (Long)":
                    action = "Sell"
            else:
                if prev_action == "No Action":
                    action = "No Action"
                else:
                    # If this pops up on the tape, it means the situation is not accounted for
                    action = "Error"
            # Only append the action for the previous day (to avoid lookahead bias)
            if next_action is not None:
                dates.append(start_date)
                actions.append(next_action) 

            prev_action = action
            next_action = action
            post_idx += 1
        
        df_actions = pd.DataFrame({'Date': dates, 'Action': actions})
        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        self.df['Buy Signal'] = np.where(self.df['Action'] == 'Buy', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Buy', self.df['Close'].shift(1), np.nan)))
        # Previous close at Sell signals
        self.df['Sell Signal'] = np.where(self.df['Action'] == 'Sell', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Sell', self.df['Close'].shift(1), np.nan)))

       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if return_table:
            print(f"Buys/Sells {df_actions['Action'].value_counts()}")
            return round(self.df,4)

    def backtest(self, print_statement=True, return_table=False, model_return=False, buy_hold=False, return_model_df=False):
        initial_investment = 10000
        cash = initial_investment
        shares = 0
        long_value = 0
        short_value = 0
        portfolio_value = []
        # Calculate Buy/Hold Value
        # previous Close is used to avoid lookahead bias
        share_cost = self.df['Previous Close'].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']
        self.df['Model Value'] = 0
        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Previous Close'].iloc[i]
            prev_action = self.df['Action'].iloc[i-1]
            if action == 'Buy':
                if prev_action == "Sell" or prev_action == "Hold (Short)":
                    cash = (shares * short_price) - ((shares * short_price) * ((price - short_price) / short_price))
                    shares = 0
                    short_value = 0
                    short_price = 0
                shares = cash/price
                long_price = price
                long_value = shares * price
                cash = 0
            elif action == 'Sell':
                if prev_action == "Buy" or prev_action == "Hold (Long)":
                    cash = (shares * long_price) + ((shares * long_price) * ((price - long_price) / long_price))
                    shares = 0
                    long_price = 0
                    long_value = 0
                short_price = price
                shares = cash/price
                short_value = shares * price
                cash = 0
            elif action == "Hold (Short)":
                short_value = (shares * short_price) - ((shares * short_price) * ((price - short_price) / short_price))
            elif action == 'Hold (Long)':
                long_value = (shares * long_price) + ((shares * long_price) * ((price - long_price) / long_price))
    
            model_value = (cash + short_value + long_value)
            portfolio_value.append(model_value)
        self.df['Model Value'] = portfolio_value

        
        if print_statement:
            print(f"{self.ticker} Buy/Hold Result: {round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)}%")
            print(f"{self.ticker} Model Result: {round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)}%")
            print(f" from {self.df.index[0]} to {self.df.index[-1]}")
        if return_table:
            return self.df
        if model_return:
            return round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)
        if buy_hold:
            return round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)
        if return_model_df:
            return self.df['Model Value']
    
    def sharpe_ratio(self, return_model=True, return_buy_hold=False):
        # factor answers the question: how many of this interval are in the total timespan
        if self.interval == "1d":
            #There are 252 trading days in a year
            annualized_factor = 252
        elif self.interval == "1wk":
            #52 weeks in a year
            annualized_factor = 52
        elif self.interval == "1mo":
            #12 months in a year
            annualized_factor = 12
        else:
            annualized_factor = 1
        model_descriptives = stats.describe(self.df['Model Value'].pct_change().dropna())
        model_mean = model_descriptives.mean
        model_std = model_descriptives.variance ** 0.5
        model_sharpe = model_mean / model_std * (annualized_factor ** 0.5)
        buy_hold_descriptives = stats.describe(self.df['Buy/Hold Value'].pct_change().dropna())
        buy_hold_mean = buy_hold_descriptives.mean
        buy_hold_std = buy_hold_descriptives.variance ** 0.5
        buy_hold_sharpe = buy_hold_mean / buy_hold_std * (annualized_factor ** 0.5)
        if return_buy_hold:
            return round(buy_hold_sharpe, 6)
        if return_model:
            return round(model_sharpe, 6)