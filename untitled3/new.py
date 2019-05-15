from __future__ import print_function
import numpy as np
from numpy import empty
import matplotlib.pyplot as plt
import pandas as pd
from model import *
import time
from math import log
from environment_new import ForEnvir
from keras.models import load_model, save_model
from keras.callbacks import TensorBoard
from random import random, randint


def barpredictor():
    inputNode = 2
    layers = 4
    outputs = 1
    layer_depth = 128
    env = ForEnvir()
    barpredictor = mlp(inputNode, outputs, layers, layer_depth)
    return barpredictor


def actFromQ(QVal,qcolumn):
    act = -1
    for i in range(0, outputs):  # find the action assosiated with this qval
        if QVal == qcolumn[i]:
            act = i  # the action associated to the maxQval
    return act


def epsilonGreed(greedChance, qcolumn):
    if greedChance >= random():  # if it is greedy pick the best
        QVal = max(qcolumn)
        act = actFromQ(QVal, qcolumn)
    else:  # else pick a random
        act = randint(0, outputs-1)
        QVal = qcolumn[act]
    #print(act,'  ', qcolumn)
    return int(act), QVal


def findQdif(Action, Discount, Alpha, CurrentQ, NextQ, Nextreward):
    # for n in range(0, len(augmented)-1):
    CurrentQVal = CurrentQ[Action]
    newQValue = CurrentQVal + (Alpha * (Nextreward + (Discount *max(NextQ.T)) - CurrentQVal))
    #print(stepvals)
    #print("act", env.Getact(Action), "rew", nextreward, "diff", newQValue - CurrentQ[Action], "Qval", CurrentQ[Action])
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    CurrentQ[Action] = newQValue
    return CurrentQ


# def test(Qnetwork, TDnet, States):
#     env.TestReset()
#     for v in range(0,env.testLen):
#         # get the Qcolumn given the state
#         Qnetwork.predict()
#         #epsilonGreed(1,)
#         #compare the best current action with the difference of the QValues
#     return 0


def train(Qnetwork,TDnet,State,Action,Reward,NextState):

    next = np.array(empty([1, inputNode]))
    update = np.array(empty([batchSize, outputs]))
    curState = np.array(empty([1, inputNode]))
    stateBatch = np.array(empty([batchSize, inputNode]))
    for x in range(0, batchSize):
        if x ==0:#learn from the current value
            num= i + p * maxItem
        else:#learn from experience
            num = randint(p * maxItem, (i + p * maxItem))  # only learn from your experience of the current episode
        next[0] = NextState.iloc[num].values
        curState[0] = State.iloc[num].values
        stateBatch[x] = curState[0]
        qVals = Qnetwork.predict(curState)
        nextq = TDnet.predict(next)
        updatedQ = findQdif(Action[num], discount, alpha, qVals[0], nextq[0], Reward[num])
        update[x] = updatedQ
    curloss = Qnetwork.fit(stateBatch, update,shuffle=False, batch_size=5,epochs=5, verbose=0)
    loss.append(sum(curloss.history['loss'])/5)

    return Qnetwork

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


env = ForEnvir()
cols =env.Getcols()

episodes = 100
alpha = 0.005
discount = 0.8
save = 1
inputNode = len(cols)
layers = 1
outputs = 4
layer_depth = 500

Qnet = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
allSave = 1
Qnet.save('untrained.h5')
target = load_model('untrained.h5')

percent = 0
finalBal = []
count = []
batchSize = 5
actions = np.array(empty([env.trainLen*episodes],int))
curQ = np.array(empty([env.trainLen*episodes, outputs]))
nextQ = np.array(empty([env.trainLen*episodes,outputs]))
nextreward = np.array(empty([env.trainLen*episodes,1]))
nextstate = np.array(empty([env.trainLen*episodes,inputNode]))
episodicLoss = []
episodicReward = []
state = pd.DataFrame(columns=cols)
nextState = pd.DataFrame(columns=cols)
maxItem = env.trainLen
for p in range(0, episodes):# for each episode

    loss = []
    print("EPISODE", p)
    # init
    greedChance = 0.7 + 0.305 * log(p + 1, episodes)
    start = time.time()

    for i in range(0, maxItem):
        #calculate percentages
        #newP =int(100*((i)/(env.trainLen)))
        #if percent != newP:
        #    print(i, '/', env.trainLen)
       # percent = newP
          #Q-learning
        rowNum = i + p * maxItem
        state = state.append(env.Getstate())  #get the current state
        curState = np.array(empty([1,inputNode]))
        curState[0] = state.iloc[rowNum].values
        qvals = Qnet.predict(curState)
        (actions[rowNum], actionQVal) = epsilonGreed(greedChance, qvals[0])#select an action

        #print('\n\n\n\n\n', env.PeakNextState(),'\n\n\n\n\n')
        nextState = nextState.append(env.PeakNextState(actions[rowNum]))

        n = np.array(empty([1, inputNode]))
        n[0] = nextState.iloc[i + p * maxItem].values

        # calculate the q target values
        nextQ[rowNum] = target.predict(n)  # get q values of the next pass

        # update the target list
        nextreward[rowNum] = env.Takeact(actions[rowNum])

        if (i)>= batchSize: ##you cant train on with less than your batch size of items
            Qnet = train(Qnet, target, state, actions, nextreward, nextState)

        env.Nextstate()
    env.Close()
    initbal = 1000000
    if p == 0:
        maxbal= env.balance-initbal
    if (env.balance > maxbal)or(allSave==1):  # if you did better than last time save
        #Qnet.save('untrained.h5')
        target.set_weights(Qnet.get_weights())
        #target = load_model('untrained.h5')
        print('PASS PROGRESS SAVED')
    else:  # if you didnt then dont save
        Qnet.set_weights(target.get_weights())
        print('FAIL PROGRESS NOT SAVED')
    totalTime = time.time() - start
    estTime = totalTime*(episodes-p)
    estTime = estTime/(60*60)
    # test(Qnet, target, env.x_test)

    Qnet.save('my_model.h5')
    finalBal.append(env.balance)
    count.append(p)
    maxbal = abs(max(finalBal))
    minbal = min(finalBal)
    episodicLoss.append(sum(loss))

    sumrew = sum(nextreward[p * maxItem: maxItem * (p+1)])
    episodicReward.append(sumrew)
    print('Episode ended with:', env.balance, '|  ETA:', round(estTime, 2), 'hours | chance:', greedChance * 100, '% | total Reward:', episodicReward[p])
    env.Resetenv()


    plt.figure(1)
    plt.subplot(311)
    plt.axis([0, episodes, minbal, maxbal])
    plt.title('account balance')
    plt.xlabel('Episodes')
    plt.ylabel('account balance')
    plt.plot(count, finalBal)
    plt.subplot(312)
    plt.xlim(0,100)
    plt.plot(count, episodicLoss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Episode')
    plt.subplot(313)
    plt.xlim(0, 100)
    plt.plot(count, episodicReward)
    plt.title('Model reward')
    plt.ylabel('reward')
    plt.xlabel('Episode')
    plt.show()
