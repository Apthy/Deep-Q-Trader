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


def epsilonGreed(greedChance, nextQ):

    if greedChance >= random():  # if it is greedy pick the best
        QVal = max(nextQ)
        for i in range(0, outputs):  # find the action assosiated with this qval

            if QVal == nextQ[i]:
                act = i  # the action associated to the maxQval
    else:  # else pick a random
        act = randint(0, outputs-1)
        QVal = nextQ[act]
    return act, QVal

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


def findQdif(action, actionQVal, discount, alpha, currentQ, nextQ):
    # for n in range(0, len(augmented)-1):
    nextreward = env.Takeact(action)
    newQValue = actionQVal + (alpha * (nextreward + (discount * max(nextQ.T)) - actionQVal))
    #print(stepvals)
    #print("act", env.Getact(action), "rew", nextreward, "diff", newQValue - currentQ[action], "Qval", currentQ[action])
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    currentQ[action] = newQValue
    return currentQ


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


episodes = 290
alpha = 1
discount = 1
save = 1
#print(env.colnums)
inputNode = 15
layers = 100
outputs = 4
layer_depth = 128
env = ForEnvir()
model = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
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

    greedChance = .8#0.9 * log(p + 1, episodes)
    for i in range(0, env.trainLen):
        #  calculate percentages
        newP =int(100*((i)/(env.trainLen)))
        if percent != newP:
            print(i, '/', env.trainLen)
        percent = newP
        #  Q-learning
        stepvals = env.Getstate()#get the current state

        #print(stepvals.shape)
        traininput = traininput.append(stepvals)
        curQprint = (model.predict(env.Getstate())[0, :])

        curQ[i] = (model.predict(env.Getstate())[0])  # get q values of the pass

        nextQ = model.predict(env.PeakNextState())[0,:]  # get q values of the next pass
        (actions, actionQVal) = epsilonGreed(greedChance, curQ[i])  #select the action you want to take

        # calculate the q target values

        # update the target list
        curQ[i] = findQdif(actions, actionQVal, discount, alpha, curQ[i], nextQ)
        env.Nextstate()
    #curQ = np.delete(curQ, le, axis=0)  # delete the final row

    # to train, take an action get a reward

    tfcb = TensorBoard(log_dir='/TensorBoardResults/'+str(p)+"/", histogram_freq=0,
                       write_graph=True, write_images=True)
    model.fit(traininput, curQ, epochs=10, callbacks=[tfcb])
    if save == 1:
        model.save('my_model.h5')
    finalBal.append(env.balance)
    count.append(p)
    print(env.balance)
    env.Resetenv()
    plt.axis([0, episodes, 990000, 1005000])
    plt.plot(count, finalBal)
    plt.show()
#once you change the course
curQ = model.predict(x_test_aug, 1)  # get the current output of the network
actionQVal = []
actions = []
env = OrderActions()
greedChance = 1
(actions, actionQVal) = epsilonGreed(greedChance, curQ) # perform entirely greedy selection to get the best q vals
curQ = findQdif(x_test_aug, actions, actionQVal, discount, alpha, curQ) # find out ohw much they are going to change

curQ = np.delete(curQ, le2, axis=0)  # delete the final row

model.evaluate(x_test, curQ)
