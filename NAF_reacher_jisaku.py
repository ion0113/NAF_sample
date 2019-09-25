# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:07:29 2019

@author: KM64864
"""

import numpy as np
import gym
import random

import keras
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation
import tensorflow as tf
from keras.optimizers import Adam
from keras.initializers import RandomUniform

from collections import deque

class Agent:
    def __init__(self, env, learning_rate=1e-3, state_dim=11, action_dim=2, hidden_size=32, test=False):#state_dim=13, action_dim=4
        self.env = env
        self.input_dim = env.observation_space.shape

        self.state_dim = state_dim
        self.actions_dim = action_dim
        
        self.testflg = test
        
        if not test:
            self.q_net_q, self.q_net_a, self.q_net_v = self.createModel()
            self.t_net_q, self.t_net_action, self.t_net_v = self.createModel()
            # 損失関数
            adam = Adam(lr=learning_rate, clipnorm=1.)
            # モデル生成
            self.q_net_q.compile(optimizer=adam, loss='mae')
            self.t_net_q.compile(optimizer=adam, loss='mae')
        else:
            self.t_net_q, self.t_net_a, self.t_net_v = self.createModel()
            self.t_net_q.load_weights("weights/weight_reacher_750.h5")

    def createModel(self):
        x = Input(shape=(self.state_dim,), name='observation_input')
        u = Input(shape=(self.actions_dim,), name='action_input')
        
        init = keras.initializers.RandomUniform(minval=-0.03, maxval=0.03, seed=None)
        
        # Middleware
        #h = Dense(64, activation="relu")(x)
        #h = Dense(64, activation="relu")(h)
        #h = Dense(32, activation="relu")(h)

        # NAF Head

        # < Value function >
        V = Dense(128, kernel_initializer=init)(x)
        V = BatchNormalization()(V)
        V = Activation("relu")(V)
        V = Dense(128, kernel_initializer=init)(V)
        V = BatchNormalization()(V)
        V = Activation("relu")(V)
        #V = Dense(32, activation="relu")(V)
        V = Dense(1, activation="linear", name='value_function')(V)

        # < Action Mean >
        mu = Dense(128, kernel_initializer=init)(x)
        mu = BatchNormalization()(mu)
        mu = Activation("relu")(mu)
        mu = Dense(128, kernel_initializer=init)(mu)
        mu = BatchNormalization()(mu)
        mu = Activation("relu")(mu)
        #mu = Dense(32, activation="relu")(mu)
        mu = Dense(self.actions_dim, activation="tanh", name='action_mean')(mu)

        # < L function -> P function >
        l0 = Dense(256, kernel_initializer=init)(x)
        l0 = BatchNormalization()(l0)
        l0 = Activation("relu")(l0)
        l0 = Dense(256, kernel_initializer=init)(x)
        l0 = BatchNormalization()(l0)
        l0 = Activation("relu")(l0)
        #l0 = Dense(64, activation="relu")(l0)
        l0 = Dense(int(self.actions_dim * (self.actions_dim + 1) / 2), kernel_initializer=init, activation="linear", name='l0')(x)
        l1 = Lambda(lambda x: tf.contrib.distributions.fill_triangular(x))(l0)
        L = Lambda(lambda x: tf.matrix_set_diag(x, tf.exp(tf.matrix_diag_part(x))))(l1)
        P = Lambda(lambda x: tf.matmul(x, tf.matrix_transpose(x)))(L)

        # < Action function >
        u_mu = keras.layers.Subtract()([u, mu])
        u_mu_P = keras.layers.Dot(axes=1)([u_mu, P]) # transpose 自動でされてた
        u_mu_P_u_mu = keras.layers.Dot(axes=1)([u_mu_P, u_mu])
        A = Lambda(lambda x: -1.0/2.0 * x)(u_mu_P_u_mu)

        # < Q function >
        Q = keras.layers.Add()([A, V])

        # Input and Output
        model_q = Model(input=[x, u], output=[Q])
        model_mu = Model(input=[x], output=[mu])
        model_v = Model(input=[x], output=[V])
        model_q.summary()
        model_mu.summary()
        model_v.summary()

        return model_q, model_mu, model_v

    def getAction(self, state):
        if not self.testflg:
            action = self.q_net_a.predict_on_batch(state[np.newaxis,:])
        else:
            action = self.t_net_a.predict_on_batch(state[np.newaxis,:])
        #action = self.q_net_a.predict_on_batch(state[np.newaxis,:])
        #action = self.env.action_space.sample()
        #print(action)
        return action

    def Train(self, x_batch, y_batch):
        return self.q_net_q.train_on_batch(x_batch, y_batch)

    def PredictT(self, x_batch):
        return self.t_net_q.predict_on_batch(x_batch)

    def WeightCopy(self):
        self.t_net_q.set_weights(self.q_net_q.get_weights())

def CreateBatch(agent, replay_memory, batch_size):
    minibatch = random.sample(replay_memory, batch_size)
    state, action, reward, state2, end_flag =  map(np.array, zip(*minibatch))

    x_batch = state
    
    next_v_values = agent.t_net_v.predict_on_batch(state2)
    y_batch = np.zeros(batch_size)

    for i in range(batch_size):
        y_batch[i] = reward[i] + 0.99 * next_v_values[i]
    return [x_batch, action], y_batch

def main():
    n_episode = 1000 # 繰り返すエピソード回数
    max_memory = 100000 # リプレイメモリの容量
    batch_size = 128

    max_sigma = 0.3 # 付与するノイズの最大分散値
    sigma = max_sigma

    reduce_sigma = max_sigma / (n_episode) # 1エピソードで下げる分散値

    #1 env = gym.make('FetchReach-v1', reward_type='dense') # 環境
    env = gym.make('Reacher-v2')
    np.random.seed(123)#(123)
    env.seed(123)#(123)
    agent = Agent(env)
    # リプレイメモリ
    replay_memory = deque()

    # ゲーム再スタート
    for episode in range(n_episode):

        #print("episode " + str(episode))
        end_flag = False
        state = env.reset()
        #state = np.hstack((state['observation'], state['desired_goal']))

        sigma -= reduce_sigma
        if sigma < 0:
            sigma = 0
            
        #prev_reward = np.inf
        sum_reward = 0

        while not end_flag:
            # 行動にノイズを付与
            noize = np.random.normal(loc=0, scale=sigma, size=2)
            #print(noize)
            action = agent.getAction(state) + noize
            #print(action)
            #action = action.reshape((4,))
            action = action.reshape((2,))
            # robot_envの行動が-1~1の範囲なので変換
            action = np.clip(action, -1.0, 1.0)
            #print(action)
            #action = np.clip(action, -1.0, 1.0)*2

            state2, reward, end_flag, info = env.step(action)
            #state2 = np.hstack((state2['observation'], state2['desired_goal']))
            
            # オリジナルの報酬計算（距離が縮まれば1, 広がれば0）
            #print('reward %f, prev_reward %f, reward < prev_reward %d' % (reward, prev_reward, reward < prev_reward))
            #if reward > prev_reward:
            #    reward_bin = 1
            #else:
            #    reward_bin = 0
            #prev_reward = reward
            sum_reward += reward
            
            # 前処理
            # リプレイメモリに保存
            replay_memory.append([state, action, reward, state2, end_flag])
            # リプレイメモリが溢れたら前から削除
            if len(replay_memory) > max_memory:
                replay_memory.popleft()
            # リプレイメモリが溜まったら学習
            if len(replay_memory) > batch_size*4:
                x_batch, y_batch = CreateBatch(agent, replay_memory, batch_size)
                agent.Train(x_batch, y_batch)

            state = state2
            # 可視化をする場合はこのコメントアウトを解除
            if episode % 10 == 0 or episode > 50:
                env.render()
        
        print('episode %03d, end_reward =%f, sum_reward=%d' % (episode, reward, sum_reward))
        
        # 4episodeに1回ターゲットネットワークに重みをコピー
        if episode != 0 and episode % 4 == 0:
            agent.WeightCopy()
            # Q-networkの重みをTarget-networkにコピー
            agent.t_net_q.save_weights("weight_reacher.h5")

    env.close()

    agent.WeightCopy()
    # Q-networkの重みをTarget-networkにコピー
    agent.t_net_q.save_weights("weight_reacher.h5")


def test():
    env = gym.make('Reacher-v2') # 環境
    agent = Agent(env, test=True)

    # ゲーム再スタート
    while 1:
        end_flag = False
        state = env.reset()

        while not end_flag:
            action = agent.getAction(state)
            action = action.reshape((2,))
            action = np.clip(action, -1.0, 1.0)

            state2, reward, end_flag, info = env.step(action)

            state = state2

            env.render()

    env.close()

if __name__ == '__main__':
    #main()
    test()