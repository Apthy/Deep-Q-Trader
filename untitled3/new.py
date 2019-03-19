from __future__ import print_function
import numpy as np
import pandas as pd
from model import *
from math import log
from environment_new import ForEnvir
from keras.callbacks import TensorBoard
from random import random, randint


def epsilonGreed(greedChance, nextQ):

    if greedChance >= random():  # if it is greedy pick the best
        QVal = max(nextQ)
        for i in range(0, 4):  # find the action assosiated with this qval
            if QVal == nextQ(i):
                act = i  # the action associated to the maxQval
    else:  # else pick a random
        act = randint(0, 3)
        QVal = nextQ[0, act]
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
    nextState = env.peakNextState()
    newendpip = nextState.close
    newstartpip = nextState.Open
    nextreward = takeAct(action, newstartpip, newendpip)
    newQ = actionQVal + (alpha * (nextreward + (discount * max(nextQ)) - actionQVal))
    print("act", getact(action),"rew", nextreward,"diff", newQ - currentQ[action], "Qval", currentQ[action])
    currentQ[action] = newQ
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

env = ForEnvir()
print(env.colnums)
inputNode = 15
layers = 20
outputs = 4
layer_depth = 32

model = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
print(env.getstate())

for p in range(0, episodes):  # for each episode
    traininput = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    curQ = [[],[],[],[]]
    print("EPISODE",p)
    # init
    for i in range(0,env.trainLen):
        stepvals = env.getstate()
        print(stepvals.shape)
        traininput[i] = stepvals
        curQ[i] = model.predict(env.getstate())  # get q values of the pass
        nextQ = model.predict(env.peakNextState())  # get q values of the pass
        greedChance = 0.9 * log(p+1, episodes)
        (actions, actionQVal) = epsilonGreed(greedChance, curQ[i])  #select the action you want to take

        # calculate the q target values

        # update the target list
        curQ[i] = findQdif(actions, actionQVal, discount, alpha, curQ[i], nextQ)

    #curQ = np.delete(curQ, le, axis=0)  # delete the final row

    # to train, take an action get a reward

    tfcb = TensorBoard(log_dir='/TensorBoardResults/'+str(p)+"/", histogram_freq=0,
                       write_graph=True, write_images=True)

    model.fit(traininput, curQ, epochs=5, callbacks=[tfcb])

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
