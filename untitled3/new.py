from __future__ import print_function
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
import pandas as pd
from model import *

from math import log
from environment_new import ForEnvir
from keras.models import load_model
from keras.callbacks import TensorBoard
from random import random, randint


def epsilonGreed(greedChance, thisQ):

    if greedChance >= random():  # if it is greedy pick the best
        QVal = max(thisQ)
        for i in range(0, outputs):  # find the action assosiated with this qval

            if QVal == thisQ[i]:
                act = i  # the action associated to the maxQval
    else:  # else pick a random
        act = randint(0, outputs-1)
        QVal = thisQ[act]
    return act, QVal


def findQdif(Action, ActionQVal, Discount, Alpha, CurrentQ, NextQ):
    # for n in range(0, len(augmented)-1):
    nextreward = env.Takeact(Action)
    newQValue = ActionQVal + (Alpha * (nextreward + (Discount * max(NextQ.T)) - ActionQVal))
    #print(stepvals)
    #print("act", env.Getact(Action), "rew", nextreward, "diff", newQValue - CurrentQ[Action], "Qval", CurrentQ[Action])
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    CurrentQ[Action] = newQValue
    return CurrentQ


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


episodes = 100
alpha = 0.1
discount = 0.7
save = 1
#print(env.colnums)
inputNode = 15
layers = 1
outputs = 4
layer_depth = 128
env = ForEnvir()
Qnet = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
Qnet.save('untrained.h5')
target = load_model('untrained.h5')

#model = load_model('my_model.h5')
#print(env.Getstate())
percent = 0
finalBal = []
count = []
fig = plt.figure()

for p in range(0, episodes):# for each episode


    traininput = pd.DataFrame(columns=['Close', 'Open', 'High', 'Low', 'SMA200', 'EMA15', 'EMA12', 'EMA26',
                    'MACD', 'Bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'StochK',
                    'StockD','isopen', 'direction'])
    curQ = np.array(empty([env.trainLen,4]))
    print("EPISODE",p)
    # init

    greedChance = 0.5 + 0.9 * log(p + 1, episodes)
    for i in range(0, env.trainLen):
        #  calculate percentages
        newP =int(100*((i)/(env.trainLen)))
        if percent != newP:
            print(i, '/', env.trainLen)
        percent = newP
        #  Q-learning
        stepvals = env.Getstate()#get the current state
        traininput = traininput.append(stepvals)
        curQ[i] = (Qnet.predict(env.Getstate()))  # get q values of the pass

        #print('\n\n\n\n\n', env.PeakNextState(),'\n\n\n\n\n')
        nextQ = target.predict(env.PeakNextState()) # get q values of the next pass
        (actions, actionQVal) = epsilonGreed(greedChance, curQ[i])  #select the action you want to take

        # calculate the q target values

        # update the target list
        curQ[i] = findQdif(actions, actionQVal, discount, alpha, curQ[i], nextQ)
        #test = np.array(curQ[i])
        outtest = np.array(empty([1, 4]))
        outtest[0] = curQ[i]
        #print(outtest.shape)

        inTest = np.array(empty([1,15]))
        inTest[0] = stepvals.iloc[0].values

        #print(inTest.shape)
        Qnet.fit(inTest,outtest, batch_size=1,epochs=1, verbose=0)
        env.Nextstate()
    #curQ = np.delete(curQ, le, axis=0)  # delete the final row
    Qnet.save('untrained.h5')
    target = load_model('untrained.h5')


    env.Close()#close any existing trades
    tfcb = TensorBoard(log_dir='/TensorBoardResults/'+str(p)+"/", histogram_freq=0,
                       write_graph=True, write_images=True)

    if save == 1:
        Qnet.save('my_model.h5')
    finalBal.append(env.balance)
    count.append(p)
    print(env.balance)
    env.Resetenv()
    plt.axis([0, episodes, 909900, 1100100])
    plt.plot(count, finalBal)
    plt.show()


#once you change the course
#curQ = Qnet.predict(x_test_aug, 1)  # get the current output of the network
#actionQVal = []
#actions = []
#env = OrderActions()
#greedChance = 1
#(actions, actionQVal) = epsilonGreed(greedChance, curQ) # perform entirely greedy selection to get the best q vals
#curQ = findQdif(x_test_aug, actions, actionQVal, discount, alpha, curQ) # find out ohw much they are going to change

#curQ = np.delete(curQ, le2, axis=0)  # delete the final row

#Qnet.evaluate(x_test, curQ)
