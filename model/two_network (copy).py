#!/usr/bin/env python

import tensorflow as tf
import cv2
import sys
import time
sys.path.append("Wrapped Game Code/")
import pong_fun as game# whichever is imported "as game" will be used
import dummy_game
import tetris_fun
import random
import numpy as np
from collections import deque

GAME = 'pong' # the name of the game being played for log files
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 200 # timesteps to observe before training
EXPLORE = 500 # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

pretrain_number=0

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    Q_max=tf.reduce_max(readout)
    tf.scalar_summary('Q_max', Q_max)

    return s, readout, h_fc1
    
def stateNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, 1])
    b_fc2 = bias_variable([1])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    state_value = tf.matmul(h_fc1, W_fc2) + b_fc2
   
    #tf.scalar_summary('state_value', state_value)

    return s, state_value

def trainNetwork(s,s_for_state, readout,state_value, h_fc1, sess,sess2):
    
    merged = tf.merge_all_summaries()
    #state_merged = tf.merge_all_summaries()
    
    writer = tf.train.SummaryWriter("log_two_net/", sess.graph)
    #state_writer = tf.train.SummaryWriter("log2/", sess2.graph)
   
    
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None,1])
    readout_action = tf.reduce_sum(tf.mul(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    #a_for_state = tf.placeholder("float", [None, ACTIONS])
    y_for_state = tf.placeholder("float", [None,1])    
    cost_for_state = tf.reduce_mean(tf.square(y_for_state - state_value))
    train_step_for_state = tf.train.AdamOptimizer(1e-6).minimize(cost_for_state)
    
    

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

    # saving and loading networks
    saver = tf.train.Saver()
    
    sess.run(tf.initialize_all_variables())
    sess2.run(tf.initialize_all_variables())
    #checkpoint = tf.train.get_checkpoint_state("new_networks/")
    
    #saver.restore(sess, "new_networks/pong-dqn-"+str(pretrain_number))
    
    
    """
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
	#saver.restore(sess, "my_networks/pong-dqn-26000")
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"
    """

    epsilon = INITIAL_EPSILON
    t = 0
    
    total_score=0
    positive_score=0    
    
    while "pigs" != "fly":
        # choose an action epsilon greedily
        #readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
        readout_t = sess.run(readout,feed_dict = {s : [s_t]})[0]
        

        a_t = np.zeros([ACTIONS])
        action_index = 0
        if random.random() <= epsilon or t <= OBSERVE:
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        for i in range(0, K):
            # run the selected action and observe next state and reward
            x_t1_col, r_t, terminal = game_state.frame_step(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                
        if r_t==1 or r_t==-1:
            total_score=total_score+r_t
        
        if r_t==1:
            positive_score=positive_score+r_t

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            
            minibatch_for_state = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]
            
#            s_j_batch_for_state = [d[0] for d in minibatch_for_state]
#            a_batch_for_state = [d[1] for d in minibatch_for_state]
#            r_batch_for_state = [d[2] for d in minibatch_for_state]
#            s_j1_batch_for_state = [d[3] for d in minibatch_for_state]

            y_batch = []
            y_batch_for_state = []
            
            #state_former_batch = state_value.eval(feed_dict = {s : s_j_batch_for_state})
            #state_later_batch = state_value.eval(feed_dict = {s : s_j1_batch_for_state})
            
            state_former_batch = sess2.run(state_value,feed_dict = {s_for_state : s_j_batch})
            state_later_batch = sess2.run(state_value,feed_dict = {s_for_state : s_j1_batch})
            
                        
            r_hat_batch = []
            for i in range(0, len(minibatch)):
                r_hat_batch.append( r_batch[i]+GAMMA*state_later_batch[i]-state_former_batch[i])      
            
                 
                   
            
            #readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            readout_j1_batch = sess.run(readout,feed_dict = {s : s_j1_batch})
            
            
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                    y_batch_for_state.append(r_batch[i])
                else:
                    y_batch.append(r_hat_batch[i]+r_batch[i]+ GAMMA * np.max(readout_j1_batch[i]))
                    y_batch_for_state.append(r_hat_batch[i] + GAMMA * state_later_batch[i])
                
            y_batch_for_state = np.reshape(y_batch_for_state,(-1,1))
            y_batch = np.reshape(y_batch,(-1,1))
            
            # perform gradient step
            sess.run(train_step,feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch})
                
            sess2.run(train_step_for_state,feed_dict = {
                    y_for_state : y_batch_for_state,                    
                    s_for_state : s_j_batch})
            
        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'two_networks/' + GAME + '-dqn', global_step = t)
            
            #saver.save(sess, 'new_networks/' + GAME + '-dqn', global_step = t)
            
        if t % 500 == 0:  
            result = sess.run(merged,feed_dict = {s : [s_t]})
            writer.add_summary(result, t)
            a_file.write(str(t+pretrain_number)+','+",".join([str(x) for x in readout_t]) + \
            ','+str(total_score)+ ','+str(positive_score)+'\n')
     
            
    
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        print "TIMESTEP", t+pretrain_number, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t)
        print total_score,positive_score
        # write info to files
        """
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            #h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            #cv2.imwrite("logs_pong/frame" + str(t) + ".png", x_t1)
        """

def playGame():
    sess = tf.Session()
    sess2 = tf.Session()

    s, readout, h_fc1 = createNetwork()
    s_for_state,state_value = stateNetwork()
    
    
    
    trainNetwork(s,s_for_state, readout,state_value, h_fc1, sess,sess2)

def main():
    playGame()

if __name__ == "__main__":
    main()
