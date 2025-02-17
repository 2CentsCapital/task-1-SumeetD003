import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import math

from TradeMaster.backtesting import Backtest, Strategy
from TradeMaster.test import GOOG
from TradeMaster.trade_management.tp_sl.atr_tm import ATR_RR_TradeManagement
from TradeMaster.risk_management.equal_weigh_rm import EqualRiskManagement
from TradeMaster.risk_management.rpt import RiskPerTrade
from TradeMaster.risk_management.volatility_atr_rm import VolatilityATR
from TradeMaster.wfo import WalkForwardOptimizer


class GaussianChannelStrategy(Strategy):
    atr_multiplier = 1.5  
    atr_period = 14
    risk_reward_ratio = 2.0
    initial_risk_per_trade = 0.1
    current_amount = 100000

    def init(self):
        """Initialize indicators and management strategies"""
        hlc3 = (self.data.High + self.data.Low + self.data.Close) / 3
        
        # Gaussian Channel Indicator
        def calculate_gaussian():
            beta = (1 - np.cos(2 * np.pi / 144)) / (np.sqrt(2) - 1)
            alpha = -beta + np.sqrt(beta ** 2 + 2 * beta)
            ema_length = int(2 / alpha - 1) if alpha != 0 else 20
            return ta.ema(pd.Series(hlc3), length=ema_length).ffill().values
        self.gaussian = self.I(calculate_gaussian)
        
        # ATR Indicator
        def calculate_atr():
            return ta.atr(pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Close), length=14).ffill().values
        self.atr = self.I(calculate_atr)
        
        # Stochastic RSI
        def calculate_stoch():
            stoch = ta.stochrsi(pd.Series(self.data.Close), length=14, rsi_length=14, k=3, d=3)
            return stoch['STOCHRSIk_14_14_3_3'].ffill().values
        self.stoch_k = self.I(calculate_stoch)
        
        # Volatility Bands
        self.high_band = self.I(lambda: self.gaussian + self.atr * 1.414)
        self.low_band = self.I(lambda: self.gaussian - self.atr * 1.414)
        
        # Initialize management strategies
        self.trade_management_strategy = ATR_RR_TradeManagement(self, risk_reward_ratio=self.risk_reward_ratio, atr_period=self.atr_period, atr_multiplier=self.atr_multiplier)
        self.risk_management_strategy = RiskPerTrade(self,initial_risk_per_trade = 0.1, profit_risk_percentage = 0.1)
        #self.risk_management_strategy = EqualRiskManagement(self, initial_risk_per_trade=self.initial_risk_per_trade)
        
        self.total_trades = len(self.closed_trades)

    def next(self):
        """Main trading logic"""
        self.on_trade_close()
        
        if len(self.data.Close) < 50:
            return

        price_above_band = self.data.Close[-1] > self.high_band[-1]
        gaussian_rising = self.gaussian[-1] > self.gaussian[-2]
        stoch_overbought = self.stoch_k[-1] > 70

        if price_above_band and gaussian_rising and stoch_overbought:
            if self.position().is_short:
                self.position().close()
            if not self.position():
                self.add_buy_trade()

        # Exit condition
        if self.position() and self.data.Close[-1] < self.gaussian[-1]:
            self.position().close()

    def add_buy_trade(self):
        """Long position entry logic"""
        risk_per_trade = self.risk_management_strategy.get_risk_per_trade()
        entry = self.data.Close[-1]
        if risk_per_trade > 0:
            stop_loss, take_profit = self.trade_management_strategy.calculate_tp_sl(direction="buy")
            stop_loss_perc = abs(entry - stop_loss) / entry
            trade_size = risk_per_trade / stop_loss_perc
            qty = math.ceil(trade_size / entry)
            self.buy(size=qty, sl=stop_loss, tp=take_profit)

    def on_trade_close(self):
        """Handle closed trades"""
        num_closed = len(self.closed_trades) - self.total_trades
        if num_closed > 0:
            for trade in self.closed_trades[-num_closed:]:
                if trade.pl < 0:
                    self.risk_management_strategy.update_after_loss()
                else:
                    self.risk_management_strategy.update_after_win()
        self.total_trades = len(self.closed_trades)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    bt = Backtest(GOOG, GaussianChannelStrategy, cash=10000, commission=.002, margin=0.01)
    stats = bt.run()
    bt.plot()
    print(stats)
    print(stats['_trades'])
    bt.tear_sheet()
    

#def constraint_function(p):
 #   """Ensure reasonable parameters during optimization"""
  #  return p['atr_multiplier'] > 1 and p['risk_reward_ratio'] > 1.5


#if __name__ == "__main__":
 #   warnings.filterwarnings("ignore", category=FutureWarning)
  #  # Define optimization parameters
   # optimization_params = {
    #    'atr_multiplier': [1.2, 1.5, 2.0],  
     #   'risk_reward_ratio': [1.5, 2.0, 2.5],  
      #  'initial_risk_per_trade': [0.05,0.1, 0.2]  
    #}
    
    # Instantiate Walk-Forward Optimizer
    #optimizer = WalkForwardOptimizer(
     #   GaussianChannelStrategy, optimization_params, constraint_function,  
      #  'Sharpe Ratio', cash=10000, commission=.002, margin=0.0015
    #)

    # Optimize using a rolling window
    #optimizer.optimize_stock('GOOG', '1day', 'us', 'firstratedata', training_candles=500, testing_candles=100)
