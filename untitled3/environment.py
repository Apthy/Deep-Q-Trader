class OrderActions:
    balance = 1000000
    startpip = 0
    endpip = 0
    direction = 0
    amount = 300
    open = 0

    def _init_(self, startOpen, startClose):
        self.balance = 1000000
        self.startpip = startOpen
        self.endpip = startClose
        self.direction = 0
        self.amount = 300
        self.open = 0

    # to close a trade
    def close(self, newstart, newendpip):
        bal = self.balance
        self.endpip = newendpip
        if self.open == 1:  # if there is a trade open
            if self.direction == 1:  # if it is going up
                self.balance = self.balance + (self.endpip/self.startpip)*self.amount  # close it and add +/- to balance
            else:
                self.balance = self.balance + (self.startpip/self.endpip)*self.amount
            self.open = 0
            reward = self.balance-bal  # dif of balance
        else:   #if there is no trade open there is no reward
            return -5
        self.updateBars(newstart, newendpip)
        return reward - 5

    # direction is a boolean of 1 being up 0 being down
    def buy(self, newstart, newendpip):
        rew =-10
        if self.open == 0:
            self.updateBars(newstart, newendpip)
            self.balance = self.balance - self.amount
            self.open = 1
            self.direction = 1
            rew = 0
        else:
            if self.direction == 0:# if it is the oposite
                rew = self.close(newstart,newendpip)
                self.updateBars(newstart, newendpip)
                self.balance = self.balance - self.amount
                self.open = 1
                self.direction = 1

        return rew

    def sell(self,newstart, newendpip):
        rew = -10
        if self.open == 0:
            self.updateBars(newstart, newendpip)
            self.balance = self.balance - self.amount
            self.open = 1
            self.direction = 0
            rew = 0
        else:# if there is a trade open
            if self.direction == 1:# if it is the oposite
                rew = self.close(newstart,newendpip)
                self.updateBars(newstart, newendpip)
                self.balance = self.balance - self.amount
                self.open = 1
                self.direction = 0
        return rew

    def hold(self, newstart, newendpip):
        self.updateBars(newstart, newendpip)
        return self.balance*(1/1000000)

    def updateBars(self, newstartPip, newendPip):
        if self.open == 1:
            self.endpip = newendPip
        else:
            self.startpip = newstartPip
            self.endpip = newendPip

