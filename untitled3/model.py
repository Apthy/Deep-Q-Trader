from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam, SGD


def mlp(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
        activation='sigmoid', loss='mse'):
    model = Sequential()
    model.add(Dense(n_neuron_per_layer, input_dim=n_obs, activation=activation))
    for _ in range(n_hidden_layer):
        model.add(Dense(n_neuron_per_layer, activation=activation))
        #model.add(Dropout(0.1))
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss=loss, optimizer=Adam(lr=0.001))
    print(model.summary())
    return model


def lstm(n_obs, n_action, n_hidden_layer=1, n_neuron_per_layer=32,
            activation='relu', loss='mse'):
    model = Sequential()
    model.add(LSTM(n_neuron_per_layer, input_shape=(n_obs, 1), return_sequences=True))

    for _ in range(n_hidden_layer):
        model.add(LSTM(n_neuron_per_layer, input_shape=(n_neuron_per_layer, 1), return_sequences=True))
    model.add(LSTM(n_neuron_per_layer, input_shape=(n_neuron_per_layer, 1)))
    model.add(Dense(n_action, activation='linear'))
    model.compile(loss=loss, optimizer=Adam())
    print(model.summary())
    return model
