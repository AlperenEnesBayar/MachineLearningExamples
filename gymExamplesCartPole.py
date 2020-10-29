import gym
import time
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.optimizers import Adam
import random
import os

env = gym.make("CartPole-v1")

class DQLAgent:
    def __init__(self, env):
        # parameter / hyperparameter
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="tanh"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")

    def load_model(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        print("Loaded model from disk")
        self.model = loaded_model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # acting: explore or exploit
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_model(episodess=100):
    if __name__ == "__main__":

        # initialize gym env and agent
        env = gym.make("CartPole-v1")
        agent = DQLAgent(env)

        batch_size = 16
        episodes = episodess

        for e in range(episodes):

            # initialize environment
            state = env.reset()

            state = np.reshape(state, [1, 4])

            time_t = 0
            while True:

                # act
                action = agent.act(state)  # select an action

                # step
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, 4])

                # remember / storage
                agent.remember(state, action, reward, next_state, done)

                # update state
                state = next_state

                # replay
                agent.replay(batch_size)

                # adjust epsilon
                agent.adaptiveEGreedy()

                time_t += 1

                if time_t % 20 == 0:
                    print(time_t)

                if done:
                    agent.save_model()
                    print("Episode: {}, time: {}".format(e, time_t))
                    break

            if time_t == 500:
                break



def test_DQL():
    agent = DQLAgent(env)
    agent.load_model()
    trained_model = agent
    state = env.reset()
    state = np.reshape(state, [1, 4])
    time_t = 0
    while True:
        env.render()
        action = trained_model.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        state = next_state
        time_t += 1
        print(time_t)
        time.sleep(0.1)
        if done:
            break
    print("Done\n")

# For Train the model comment test_DQL part in first time.
train_model(100)

#When model trained uncomment this and comment train_model() part for testing model.
# while True:
#     test_DQL()
