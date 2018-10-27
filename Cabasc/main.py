# ─── MODEL A ────────────────────────────────────────────────────────────────────

import tensorflow as tf
import pandas as pd


#ASPECT RECOGNITION
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



# ─── PRE-PROCESSING BOOK IMPLEMENTATION ──────────────────────────────────────────────────────────────────────────







# ─── 1 TAKE INPUT ───────────────────────────────────────────────────────────────

# how to shape the input???
# well input is a R (d* V)
d = 128 #dimension of vector
vocab = 20000 #number of vocabulary words




"""
0. How to cope with changes in sentence lenght? (how my model is going to take different input size????)
1. What's None in the placeholder shape?
2. how to initialize the word embedding vector?
3. How to convert each word to its one-hot vector?"""
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (d, vocab))
    y = tf.placeholder(tf.float32, (None, ))








# ─── 2 FORM THE MEMORY ──────────────────────────────────────────────────────────

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