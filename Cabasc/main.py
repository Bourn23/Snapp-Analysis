"""
In Progress:
1. GRU  , left & right unit
"""

"""
Must fix:
1. whole_value (which is the array of the whole word embedding) so I can convert it to a matrix
2. Change GRU's loading direction adapted to persian
"""

import tensorflow as tf
import pandas as pd
import numpy as np

# ─── 0 INPUT PRE-PROCESSING ─────────────────────────────────────────────────────────
"""word embedding algorithm"""
#i will be using twitter hashtags

# __ VARIABLE __
E = [] #vector E, consists of e, which are computed one-hots times L, L is the sentence?
#load data
data = pd.load_csv("twitter_hashtag.csv")
#load each row of sentence
for row in data.row:
    #parse the tweet, have the number of words in it
    #shape the sentence, word embeddings?!
    
    # compute one-hot vector for each word in the sentence
    for word in data['words']: #words column consists of the sentence word in a list!
        data['one-hot'] = word.compute_one_hot()
        E.append(data['one-hot'] * L) #what's L? how to multiply it?

    #processing the aspect and taking into account Va (embedding of aspect's vector)
    if len(data['aspect'] == 1):
        #take its e, as Va
        V_a = E[data['aspect_loc']]
    else: #if aspect is more than one word
        V_a = np.mean([E[i] for i in data['aspect_loc']]) #make a list of aspect's e and then compute the mean



whole_value = np.ndarray() #arr of the whole word embedding vocab
# 1 TAKE INPUT ───────────────────────────────────────────────────────────────

# how to shape the input???
# well input is a R (d* V)
d_size = 100 #dimension of word embedding's vector
vocab_size = 286000 #number of words in vocabulary

"""
0. How to cope with changes in sentence lenght? (how my model is going to take different input size????)
1. What's None in the placeholder shape?
2. how to initialize the word embedding vector?
3. How to convert each word to its one-hot vector?"""

L = tf.constant(tf.convert_to_tensor(whole_value, dtype=tf.float32), dtype=tf.float32, shape=(d_size, vocab_size), name="L")
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (d_size, vocab_size)) #what's the shape of the input??
    y = tf.placeholder(tf.float32, ([], )) #shape of y input is a scalar



# 2 CAM Memory Module ──────────────────────────────────────────────────────────
"""
0. Let's embed one-hot in input?
1. Which is faster to obtain an element from the Vocabulary matrix?
    1. do V * One-Hot 
    2. Retrieve from V
    ** For now we go with the 1st solution"""

# ── GRU UNIT ──

def gru(left_input, right_input):
    with tf.variable_scope("gru_left"):
        r_t = tf.nn.sigmoid()


def cam(input,  aspect_loc):
    #aspect_loc = [start_loc, end_loc]
    #input the one-hot vector for the 1st sentence
    """Cabasc block"""
    with tf.variable_scope("input"):
        #taking apart the input into 2 sequences and get their e's
        e = [tf.matmul(L * i) for i in input] 
        #one_hot for left
        e_ls = e[:aspect_loc[-1]] #till the last aspect

        #one_hot for right
        e_rs = e[aspect_loc[0]:] #from first aspect






#__VARIABLE__
M = [] #the memory matrix
tf.Variable()
"""
0. How to define the Memory Matrix? 
1. ONLY in tensorflow!
"""











#3 Content attention module

#4 MLP

#5 Softmax