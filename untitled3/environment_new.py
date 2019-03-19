import numpy as np
import pandas as pd


class ForEnvir:

    def __init__(self):

        path = "data/GBPUSD.csv"
        colnames = ['close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
                    'MACD', 'bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'stochK',
                    'stockD']
        self.data = pd.read_csv(path, names=colnames)  # load data
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
        self.amount = 300
        self.isopen = 0

        self.close = 0
        self.open = 0
        self.high = 0
        self.low = 0
        self.SMA200 = 0
        self.EMA15 = 0
        self.EMA12 = 0
        self.EMA26 = 0
        self.MACD = 0
        self.bollingerbandupper3sd200 = 0
        self.bollingerbandlower3sd200 = 0
        self.stochK = 0
        self.stockD = 0

    # to close a trade
    def close(self, newstart, newendpip):
        bal = self.balance
        self.endpip = newendpip
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                self.balance = self.balance + (self.endpip/self.startpip)*self.amount  # close it and add +/- to balance
            else:
                self.balance = self.balance + (self.startpip/self.endpip)*self.amount
            self.isopen = 0
            reward = self.balance-bal  # dif of balance
        else:   #if there is no trade open there is no reward
            return -5
        self.updateBars(newstart, newendpip)
        return reward - 5

    # direction is a boolean of 1 being up 0 being down
    def buy(self, newstart, newendpip):
        rew =-10
        if self.isopen == 0:
            self.updateBars(newstart, newendpip)
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 1
            rew = 0
        else:
            if self.direction == 0:# if it is the oposite
                rew = self.close(newstart,newendpip)
                self.updateBars(newstart, newendpip)
                self.balance = self.balance - self.amount
                self.isopen = 1
                self.direction = 1

        return rew

    def sell(self,newstart, newendpip):
        rew = -10
        if self.isopen == 0:
            self.updateBars(newstart, newendpip)
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 0
            rew = 0
        else:# if there is a trade open
            if self.direction == 1:# if it is the oposite
                rew = self.close(newstart,newendpip)
                self.updateBars(newstart, newendpip)
                self.balance = self.balance - self.amount
                self.isopen = 1
                self.direction = 0
        return rew

    def hold(self, newstart, newendpip):
        self.updateBars(newstart, newendpip)
        return self.balance*(1/1000000)

    def updatebars(self, newstartPip, newendPip):
        if self.isopen == 1:
            self.endpip = newendPip
        else:
            self.startpip = newstartPip
            self.endpip = newendPip

    def getact(self, action):
        if action == 0:
            rew = "sell"
        elif action == 1:
            rew = "buy"
        elif action == 2:
            rew = "hold"
        elif action == 3:
            rew = "close"
        return rew

    def takeact(self, action, newStart, newEnd):
        if action == 0:
            rew = self.sell(newStart, newEnd)
        elif action == 1:
            rew = self.buy(newStart, newEnd)
        elif action == 2:
            rew = self.hold(newStart, newEnd)
        elif action == 3:
            rew = self.close(newStart, newEnd)
        return rew

    def getstate(self):
        retval = self.data.iloc[self.currentStep]
        retval = retval.append(pd.Series([self.isopen, self.direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T

    def nextstate(self, action):
        self.currentStep += 1

    def peakNextState(self):
        retval = self.data.iloc[self.currentStep + 1]
        retval = retval.append(pd.Series([self.isopen, self.direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T
