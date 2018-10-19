from collections import deque
import time

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam

import gym
from splix_online_env import SplixOnlineEnv


# Adapted from https://gist.github.com/yashpatel5400/049fe6f4372b16bab5d3dab36854f262#file-mountaincar-py

class DQN:
    def __init__(self, env, reuse_model=False):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()
        if reuse_model:
            self.model.load_weights("model.h5")
            self.target_model.load_weights("target_model.h5")

    def create_model(self):
        model = Sequential()
        model.add(
            Conv2D(24, 7, activation="relu", data_format="channels_last", input_shape=self.env.observation_space.shape))
        model.add(Conv2D(24, 7, activation="relu", data_format="channels_last"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state.reshape(1, 51, 51, 6)))

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state.reshape(1, 51, 51, 6))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state.reshape(1, 51, 51, 6))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state.reshape(1, 51, 51, 6), target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self):
        self.model.save_weights("model.h5")
        self.target_model.save_weights("target_model.h5")


def main():
    env = SplixOnlineEnv()

    trials = 1000
    trial_len = 500

    dqn_agent = DQN(env=env)

    for trial in range(trials):
        cur_state = env.reset()
        start = time.time()
        for step in range(trial_len):

            action = dqn_agent.act(cur_state)
            new_state, reward, done, info = env.step(action)

            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        print(f"Trial {trial} : final score {info['score']} in {step+1} steps ({duration/(step+1):0.3f}s per step)")
        dqn_agent.save_model()


if __name__ == "__main__":
    main()
