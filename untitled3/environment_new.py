import numpy as np
import pandas as pd
from sklearn import preprocessing

class ForEnvir:

    def __init__(self):

        path = "data/sineTest.csv"
        #self.colnames = ['Close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
        #            'MACD', 'Bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'StochK',
        #            'StockD']
        self.colnames = ['Close', 'Open']
        self.data = pd.read_csv(path, names=self.colnames)  # load data
        #self.data.drop(labels=['High','Low', 'EMA15', 'EMA12', 'EMA26'], axis=1, inplace=True)
        self.length, self.colnums = self.data.shape
        self.netinputs = self.NormaliseData()
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
        self.direction = -1
        self.amount = 100
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
        dir = self.direction
        #self.endpip = self.data.Close[self.currentStep]
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                dif = self.endpip - self.startpip
                self.balance = self.balance + ((1+dif) * self.amount)  # close it and add +/- to balance
            else:  # going down
                dif = self.startpip-self.endpip
                self.balance = self.balance + ((1+dif) * self.amount)
            self.isopen = 0
            reward = (dif)*self.amount  # dif of balance
            self.direction=-1
        else:  # if there is no trade open there is no reward
            return -1
        #print('Direction:', dir, ' P/L:', round((dif)*self.amount,5),' Reward:',round(reward/5,5), 'difference',round(dif,5),' balance:', round(self.balance,5))
        return reward/self.amount


    def GetTradeVal(self,curstep):
        bal = self.balance
        endpip = self.data.Close[curstep]
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                dif = endpip - self.startpip
            else:  # going down
                dif = self.startpip - endpip
            reward = (1+dif) * self.amount  # change of balance
        else:  # if there is no trade open there is no reward
            return -1
        return reward/self.amount


    # direction is a boolean of 1 being up 0 being down
    def Buy(self):
        rew = -0.1
        if self.isopen == 0:  # no trade open
            self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 1
            rew = 0.1
        #else:  # trade open
        #    if self.direction == 0:  # if it is the oposite
        #        rew = self.Close()
        #        self.Updatebars()
        #        self.balance = self.balance - self.amount
        #        self.isopen = 1
        #        self.direction = 1
        return rew

    def Sell(self):
        rew = -0.1
        if self.isopen == 0:
            self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 0
            rew = 0.1
        #else:  # if there is a trade open
        #
        #    if self.direction == 1:  # if it is the opposite
        #        rew = self.Close()
        #        self.Updatebars()
        #        self.balance = self.balance - self.amount
        #        self.isopen = 1
        #        self.direction = 0
        return rew

    def Hold(self):
        return 0

    def TestSell(self):
        if self.isopen == 0:
            newisopen = 1
            newdirection = 0
        else:
            newisopen = self.isopen
            newdirection = self.direction
        return newdirection, newisopen

    def TestBuy(self):
        if self.isopen == 0:  # no trade
            newisopen = 1
            newdirection = 1
        else:
            newisopen = self.isopen
            newdirection = self.direction
        return newdirection, newisopen

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
        retval = self.netinputs.iloc[self.currentStep]
        retval = retval.append(pd.Series([self.isopen, self.direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T

    def Nextstate(self):
        self.currentStep += 1
        self.Updatebars()

    def PeakNextState(self, action):
        if(action==0):#sell
            [direction,isopen]=self.TestSell()
        elif action==1:#buy
            [direction,isopen]=self.TestBuy()
        elif action == 2:#hold
            direction = self.direction
            isopen = self.isopen
        else: #close
            isopen = 0
            direction = -1

        retval = self.netinputs.iloc[self.currentStep + 1]
        retval = retval.append(pd.Series([isopen, direction]))
        retval = retval.rename({0: 'isopen', 1: 'direction'})
        return retval.to_frame().T  # transposed frame as it was the wrong way when it was converted to a series

    def Resetenv(self):
        self.currentStep = 0
        self.balance = 1000000
        self.isopen = 0
        self.direction = 0
        self.Updatebars()

    def NormaliseData(self):
        df = pd.DataFrame(None, None, self.data.columns)
        data = pd.DataFrame(preprocessing.normalize(self.data, 'max', 0), None, self.data.columns)
        # l1 shrinks the less important features, l2 prevents overfitting
        return data


