from __future__ import print_function
import numpy as np
import pandas as pd
from model import *
from math import log
from environment_new import ForEnvir
from keras.callbacks import TensorBoard
from random import random, randint


def epsilonGreed(greedChance, nextQ):
    actionQVal=[]
    actions=[]
    for n in nextQ:  # find the max value or a random value
        q = 0
        act = 0
        if greedChance >= random():
            q = max(n)
            actionQVal.append(max(n))
        else:
            act = randint(0, 3)
            q = n[act]
            actionQVal.append(q)
        for i in range(0, 4):
            if q == n[i]:
                actions.append(i)  # the action associated to the maxQval
    return actions, actionQVal

#def epsilonGreed(greedChance, nextQ):
#    actionQVal=[]
#    actions=[]
#    for n in nextQ:  # find the max value or a random value
#        q = 0
#        act = 0
#        if greedChance >= random():
#            q = max(n)
#            actionQVal.append(max(n))
#        else:
#            act = randint(0, 3)
#            q = n[act]
#            actionQVal.append(q)
#        for i in range(0, 4):
#            if q == n[i]:
#                actions.append(i)  # the action associated to the maxQval
#    return actions, actionQVal

# calculate the forward training values by using the next action as the next state
def findQdif(augmented, actions, actionQVal, discount, alpha, nextQ):
    for n in range(0, len(augmented)-1):
        newendpip = augmented.close[n + 1]
        newstartpip = augmented.Open[n + 1]
        nextreward = takeAct(actions[n], newstartpip, newendpip)
        newQ = actionQVal[n] + (alpha * (nextreward + (discount * max(nextQ[n + 1])) - actionQVal[n]))
        print("act",getact(actions[n]),"rew", nextreward,"diff", newQ - nextQ[n, actions[n]],"Qval", nextQ[n, actions[n]])
        nextQ[n, actions[n]] = newQ
    return nextQ

def getact(action):
    if action == 0:
        rew = "sell"
    elif action == 1:
        rew = "buy"
    elif action == 2:
        rew = "hold"
    elif action == 3:
        rew = "close"
    return rew

def takeAct(action, newStart, newEnd):
    if action == 0:
        rew = env.sell(newStart, newEnd)
    elif action == 1:
        rew = env.buy(newStart, newEnd)
    elif action == 2:
        rew = env.hold(newStart, newEnd)
    elif action == 3:
        rew = env.close(newStart, newEnd)
    return rew


episodes = 90
alpha = 0.2
discount = 1

# path = "data/GBPUSD.csv"
# colnames = ['close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
#             'MACD', 'bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'stochK',
#             'stockD']
#data = pd.read_csv(path, names=colnames)  # load data
env = ForEnvir()
print(env.colnums)
#maxlen = len(data)
#
#(trainNum, null) = divmod(maxlen * 8, 10)#divide the data into training and test sets
#testNum = maxlen - trainNum
#
#x_train = data.iloc[
#          0:trainNum]  # split it into 2 data sets to test on and prevent over-fitting
#x_trainAug = data.iloc[
#             0:trainNum + 1]  # split it into 2 data sets to test on and prevent over-fitting
#x_test = data.iloc[trainNum:maxlen-1]
#x_test_aug = data.iloc[trainNum:maxlen]
#le = len(x_train)
#le2 = len(x_test)
inputNode = 15
layers = 20
outputs = 4
layer_depth = 32

model = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
#ForEnvir.setData(x_trainAug)
print(env.getstate())


for p in range(0, episodes):  # for each episode
    print("EPISODE",p)
    #init
    actionQVal = []
    actions = []
    stepvals = env.getstate()
    print(stepvals.shape)
    nextQ = model.predict(env.getstate())  # get q values of the pass
    greedChance = 0.9    * log(p+1, episodes)
    (actions, actionQVal) = epsilonGreed(greedChance, nextQ)

    # calculate the q target values

    # update the target list
    nextQ = findQdif(x_trainAug, actions, actionQVal, discount, alpha, nextQ)
    nextQ = np.delete(nextQ, le, axis=0)  # delete the final row

    # to train, take an action get a reward

    tfcb = TensorBoard(log_dir='/TensorBoardResults/'+str(p)+"/", histogram_freq=0,
                       write_graph=True, write_images=True)

    model.fit(x_train, nextQ, epochs=5, callbacks=[tfcb])

#once you change the course
nextQ = model.predict(x_test_aug, 1)  # get the current output of the network
actionQVal = []
actions = []
env = OrderActions()
greedChance = 1
(actions, actionQVal) = epsilonGreed(greedChance, nextQ) # perform entirely greedy selection to get the best q vals
nextQ = findQdif(x_test_aug, actions, actionQVal, discount, alpha, nextQ) # find out ohw much they are going to change

nextQ = np.delete(nextQ, le2, axis=0)  # delete the final row

model.evaluate(x_test, nextQ)
