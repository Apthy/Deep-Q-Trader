import numpy as np
import pandas as pd
from sklearn import preprocessing

class ForEnvir:

    def __init__(self):

        path = "data/GBPUSD.csv"
        self.colnames = ['Close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
                    'MACD', 'Bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'StochK',
                    'StockD']

        self.data = pd.read_csv(path, names=self.colnames)  # load data
        self.data.drop(labels=['High','Low', 'EMA15', 'EMA12', 'EMA26'], axis=1, inplace=True)
        self.length, self.colnums = self.data.shape
        self.netinputs = self.NormaliseData()
        self.currentStep = 0

        (trainNum, null) = divmod(self.length * 8, 10)  # divide the data into training and test sets

        self.x_train = self.netinputs.iloc[0:trainNum]  # split it into 2 data sets to test on and prevent over-fitting
        self.x_trainAug = self.netinputs.iloc[0:trainNum + 1]

        self.x_test = self.netinputs.iloc[trainNum:self.length - 1]
        self.x_test_aug = self.netinputs.iloc[trainNum:self.length]
        self.trainLen = len(self.x_train)
        self.testLen = len(self.x_test)

        self.balance = 1000000
        self.startpip = 0.00
        self.endpip = 0.00
        self.direction = -1
        self.amount = 1000
        self.isopen = 0
        self.Updatebars()

    # to close a trade
    def Close(self):
        self.Updatebars()
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                dif = self.endpip - self.startpip
                self.balance = self.balance + ((1+dif) * self.amount)  # close it and add +/- to balance
            else:  # going down
                dif = self.startpip-self.endpip
                self.balance = self.balance + ((1+dif) * self.amount)
            self.isopen = 0
            reward = dif  # dif of pips
            self.direction= -1
        else:  # if there is no trade open there is no reward
            reward = -0.001
        #print('Direction:', dir, ' P/L:', round((dif)*self.amount,5),' Reward:',round(reward/5,5), 'difference',round(dif,5),' balance:', round(self.balance,5))
        return reward

    def GetTradeVal(self,curstep):
        endpip = self.data.Close[curstep]
        if self.isopen == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                dif = endpip - self.startpip
            else:  # going down
                dif = self.startpip - endpip
            reward = (dif) * self.amount  # change of balance
        else:  # if there is no trade open there is no reward
            return -0.001
        return reward/self.amount

    # direction is a boolean of 1 being up 0 being down
    def Buy(self):

        if self.isopen == 0:  # no trade open
            self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 1
            rew = 0.001
        else:  # trade open
            # if self.direction == 0:  # if it is the oposite
            #     rew = self.Close()
            #     self.Updatebars()
            #     self.balance = self.balance - self.amount
            #     self.isopen = 1
            #     self.direction = 1
            # else:
            rew = -0.001
        return rew

    def Sell(self):
        rew = 0
        if self.isopen == 0:
            self.Updatebars()
            self.balance = self.balance - self.amount
            self.isopen = 1
            self.direction = 0
            rew = 0.001
        else:  # if there is a trade open
            # if self.direction == 1:  # if it is the opposite
            #     rew = self.Close()
            #     self.Updatebars()
            #     self.balance = self.balance - self.amount
            #     self.isopen = 1
            #     self.direction = 0
            # else:
            rew = -0.001
        return rew

    def Hold(self):
        rew = np.tanh(self.GetTradeVal(self.currentStep))*0.001
        return rew

    def TestSell(self):
        if self.isopen == 0:
            newdirection = 0
        else:
            newdirection = self.direction
        return newdirection

    def TestBuy(self):
        if self.isopen ==0:
            newdirection = 1
        else:
            newdirection = self.direction
        return newdirection

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
        else:
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
        retval = retval.append(pd.Series([self.direction]))
        retval = retval.rename({0: 'direction'})
        return retval.to_frame().T

    def Nextstate(self):
        self.currentStep += 1
        self.Updatebars()

    def PeakNextState(self, action):
        if(action==0):  # sell
            direction=self.TestSell()
        elif action==1:  # buy
            direction=self.TestBuy()
        elif action == 2:  # hold
            direction = self.direction
        else:  # close
            direction = -1

        retval = self.netinputs.iloc[self.currentStep + 1]
        retval = retval.append(pd.Series([direction]))
        retval = retval.rename({0: 'direction'})
        return retval.to_frame().T  # transposed frame as it was the wrong way when it was converted to a series

    def Resetenv(self):
        self.currentStep = 0
        self.balance = 1000000
        self.isopen = 0
        self.direction = -1
        self.Updatebars()

    def NormaliseData(self):
        df = pd.DataFrame(None, None, self.data.columns)
        data = pd.DataFrame(preprocessing.normalize(self.data, 'max', 0), None, self.data.columns)
        # l1 shrinks the less important features, l2 prevents overfitting
        return data

    def Getcols(self):
        state = self.Getstate()
        cols = state.columns.values
        return cols



