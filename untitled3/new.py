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
    return act, QVal


def findQdif(Action, CurrentQVal, Discount, Alpha, CurrentQ, NextQ):
    # for n in range(0, len(augmented)-1):
    nextreward = env.Takeact(Action)
    nextact = actFromQ(max(NextQ.T), NextQ[0,:])
    if (nextact == 3) & (env.isopen == 1):
        futureReward = env.GetTradeVal(env.currentStep+1)
    elif (nextact == 2):
        futureReward = env.GetTradeVal(env.currentStep+1)*0.1
    else:
        futureReward = 0
    newQValue = CurrentQVal + (Alpha * (nextreward + (Discount *futureReward) - CurrentQVal))
    #print(stepvals)
    #print("act", env.Getact(Action), "rew", nextreward, "diff", newQValue - CurrentQ[Action], "Qval", CurrentQ[Action])
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    CurrentQ[int(Action)] = newQValue
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
alpha = 0.0005
discount = 0.3
save = 1
#print(env.colnums)
inputNode = 4
layers = 16
outputs = 4
layer_depth = 128
env = ForEnvir()
Qnet = mlp(inputNode, outputs, layers, layer_depth)  # make an mlp
allSave = 1
Qnet.save('untrained.h5')
target = load_model('untrained.h5')

#bp = barpredictor()

#model = load_model('my_model.h5')
#print(env.Getstate())
percent = 0
finalBal = []
count = []
#trainLen =100
#predinput = env.netinputs['Open'].drop(env.length-1)[0:trainLen].values
#predinput = [predinput, env.netinputs['Close'].drop(env.length-1)[0:trainLen].values]

#predinput = predinput.drop(env.length-1)
#predoutput = env.netinputs['Close'].drop(0).values

#vals =predinput[0:trainLen,0:trainLen]
#bp.fit(predinput,predoutput[0:trainLen],epochs=10,validation_split=0.8)
#print(bp.evaluate(predinputopen[trainLen:len(predoutput)],predoutput[trainLen:len(predoutput)]))
#for i in range(trainLen,len(predinputopen)):
#    inp = predinputopen[i:i+1]
    #bp.predict(inp.values)#
#    print(bp.predict(inp.values),predoutput[i:i+1].values)

for p in range(0, episodes):# for each episode

    #colnums = env.data.columns.values
    #colnums = np.append(colnums,['isopen','direction'])
    #colnums = pd.Series(colnums)
    #traininput = pd.DataFrame(columns=['Close', 'Open', 'SMA200',
    #                'MACD', 'Bollinger_band_upper_3sd_200', 'bollinger_band_lower_3sd_200', 'StochK',
    #                'StockD','isopen','direction'])
    traininput = pd.DataFrame(columns=['Close', 'Open', 'isopen', 'direction'])
    curQ = np.array(empty([env.trainLen,outputs]))
    print("EPISODE",p)
    # init
    actions = np.array(empty([env.trainLen]))
    greedChance = 0.7 + 0.2 * log(p + 1, episodes)
    start = time.time()

    for i in range(0, 200):#env.trainLen):
        #calculate percentages
        #newP =int(100*((i)/(env.trainLen)))
        #if percent != newP:
        #    print(i, '/', env.trainLen)
       # percent = newP
          #Q-learning
        stepvals = env.Getstate()#get the current state
        traininput = traininput.append(stepvals)
        curQ[i] = (Qnet.predict(env.Getstate()))  # get q values of the pass

        #print('\n\n\n\n\n', env.PeakNextState(),'\n\n\n\n\n')


        (actions[i], actionQVal) = epsilonGreed(greedChance, curQ[i])
        # calculate the q target values
        nextQ = target.predict(env.PeakNextState(actions[i]))  # get q values of the next pass
        # update the target list
        curQ[i] = findQdif(actions[i], actionQVal, discount, alpha, curQ[i], nextQ)

        outtest = np.array(empty([1, outputs]))
        outtest[0] = curQ[i]

        inTest = np.array(empty([1,inputNode]))
        inTest[0] = stepvals.iloc[0].values

        #print(inTest.shape)
        Qnet.fit(inTest,outtest, batch_size=1,epochs=1, verbose=0)
        env.Nextstate()
    #curQ = np.delete(curQ, le, axis=0)  # delete the final row
    env.Close()
    initbal = 1000000
    if p == 0:
        maxbal= env.balance-initbal
    if (env.balance > maxbal)or(allSave==1):  # if you did better than last time save
        #Qnet.save('untrained.h5')
        target.set_weights(Qnet.get_weights())
        #target = load_model('untrained.h5')
        print('PASS PROGRESS SAVED')
    else:  # if you diddnt then dont save
        Qnet.set_weights(target.get_weights())
        print('FAIL PROGRESS NOT SAVED')
    totalTime = time.time() - start
    estTime = totalTime*(episodes-p)
    estTime = estTime/(60*60)

    tfcb = TensorBoard(log_dir='/TensorBoardResults/'+str(p)+"/", histogram_freq=0,
                       write_graph=True, write_images=True)

    if save == 1:
        Qnet.save('my_model.h5')
    finalBal.append(env.balance)
    count.append(p)
    print('Episode ended with:', env.balance, '|  ETA:', round(estTime, 2), 'hours | chance:', greedChance)
    env.Resetenv()
    maxbal = abs(max(finalBal))
    minbal = min(finalBal)

    if (p+1) % 10 == 0:
        plt.axis([0, episodes, minbal,maxbal])
        plt.xlabel('episodes')
        plt.ylabel('account balance')
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


