import numpy as np
import pandas as pd


class ForEnvir:

    def __init__(self):

        path = "data/GBPUSD.csv"
        self.colnames = ['Close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
                    'MACD', 'Bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'StochK',
                    'StockD']
        self.data = pd.read_csv(path, names=self.colnames)  # load data
        self.length, self.colnums = self.data.shape
        self.currentStep = 0

        (trainNum, null) = divmod(self.length * 8, 10)  # divide the data into training and test sets

        self.x_train = self.data.iloc[0:trainNum]  # split it into 2 data sets to test on and prevent over-fitting
        self.x_trainAug = self.data.iloc[
                          0:trainNum + 1]  # split it into 2 data sets to test on and prevent over-fitting
        self.x_test = self.data.iloc[trainNum:self.length - 1]
        self.x_test_aug = self.data.iloc[trainNum:self.length]
        self.trainLen = len(self.x_train)
        self.testLen = len(self.x_test)

        self.balance = 1000000
        self.startpip = 0.00
        self.endpip = 0.00
        self.direction = 0
        self.amount = 1000
        self.isopen = 0
        self.Updatebars()
        # self.close = 0
        # self.open = 0
        # self.high = 0
        # self.low = 0
        # self.sma200 = 0
        # self.ema15 = 0
        # self.ema12 = 0
        # self.ema26 = 0
        # self.macd = 0
        # self.bollingerbandupper3sd200 = 0
        # self.bollingerbandlower3sd200 = 0
        # self.stochK = 0
        # self.stockD = 0

    #
    # to close a trade
    def Close(self):
        bal = self.balance
        #self.endpip = self.data.Close[self.currentStep]
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                self.balance = self.balance + (
                        self.endpip / self.startpip) * self.amount  # close it and add +/- to balance
            else:  # going down
                self.balance = self.balance + (self.startpip / self.endpip) * self.amount
            self.isopen = 0
            reward = (self.balance - bal) - self.amount  # dif of balance

        else:  # if there is no trade open there is no reward
            return -5
        #print('Direction:', self.direction, ' difference:', reward, '  open:', self.startpip, ' close:', self.endpip)
        #self.Updatebars()
        return reward - 5

    # direction is a boolean of 1 being up 0 being down
    def Buy(self):
        rew = -1
        if self.isopen == 0:  # no trade open
            #self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 1
            rew = 0
        else:  # trade open
            if self.direction == 0:  # if it is the oposite
                rew = self.Close()
                self.Updatebars()
                self.balance = self.balance - self.amount
                self.isopen = 1
                self.direction = 1
        return rew

    def Sell(self):
        rew = -1
        if self.isopen == 0:
            #self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 0
            rew = 0
        else:  # if there is a trade open
            if self.direction == 1:  # if it is the opposite
                rew = self.Close()
                self.Updatebars()
                self.balance = self.balance - self.amount
                self.isopen = 1
                self.direction = 0
        return rew

    def Hold(self):
        return 0

    def Updatebars(self):
        if self.isopen == 1:
            self.endpip = self.data.Close[self.currentStep]
        else:
            self.startpip = self.data.Open[self.currentStep]
            self.endpip = self.data.Close[self.currentStep]

    def Getact(self, action):
        if action == 0:
            actName = "sell"
        elif action == 1:
            actName = "buy"
        elif action == 2:
            actName = "hold"
        elif action == 3:
            actName = "close"
        return actName

    def Takeact(self, action):
        if action == 0:
            rew = self.Sell()
        elif action == 1:
            rew = self.Buy()
        elif action == 2:
            rew = self.Hold()
        elif action == 3:
            rew = self.Close()
        return rew

    def Getstate(self):
        retval = self.data.iloc[self.currentStep]
        retval = retval.append(pd.Series([self.isopen, self.direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T

    def Nextstate(self):
        self.currentStep += 1
        self.Updatebars()

    def PeakNextState(self):
        retval = self.data.iloc[self.currentStep + 1]
        retval = retval.append(pd.Series([self.isopen, self.direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T

    def Resetenv(self):
        self.currentStep = 0
        self.balance = 1000000
        self.isopen = 0
        self.Updatebars()
