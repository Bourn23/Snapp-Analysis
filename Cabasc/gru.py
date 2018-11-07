import tensorflow as tf
class GRU_Cell(object):
    __init__(self, input_nodes, hidden_nodes, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        """ 
        0. Forgot self.W_r , self.U_r...!
        1. What are the difference between float32 & float64 ? float64: double precision, slower computation & stores very big/small numbers
        2. Do I need a output layer too? Is there an existing one? What it will be?
       ! 3. Something might go wrong with DIMENSIONS (matrix multiplication) in gru r, z !
        """
        #weights initialization for W's
        weight_init = tf.truncated_normal_initializer(mean = 1. , stddev= .01)
        self.W_r = tf.get_variable(name = "Wr", shape = [self.input_nodes, self.hidden_nodes], dtype = tf.float32, initializer=weight_init)
        self.W_z = tf.get_variable(name = "Wz", shape = [self.hidden_nodes, self.hidden_nodes], dtype = tf.float32, initializer=weight_init)
        self.W_h = tf.get_variable(name = "Wh", shape = [self.hidden_nodes, self.output_nodes], dtype = tf.float32, initializer=weight_init)

        #bias initialization
        # self.U_r = tf.get_variable(name="Ur", shape = [self.hidden_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))
        # self.U_z = tf.get_variable(name="Uz", shape = [self.hidden_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))
        # self.U_h = tf.get_variable(name="Uh", shape = [self.output_nodes], initializer=tf.zeros_initializer(dtype=tf.float32))

        #weights initialization for U's
        self.U_r = tf.get_variable(name = "Ur", shape = [self.input_nodes, self.hidden_nodes], dtype = tf.float32, initializer=weight_init)
        self.U_z = tf.get_variable(name = "Uz", shape = [self.hidden_nodes, self.hidden_nodes], dtype = tf.float32, initializer=weight_init)
        self.U_h = tf.get_variable(name = "Uh", shape = [self.hidden_nodes, self.output_nodes], dtype = tf.float32, initializer=weight_init)

        #inputs in format of [batch_size, sequence, embeddings]
        self._inputs = tf.placeholder(dtype = tf.float32, shape = [None, None, self.input_nodes], name = "GRU inputs")

        #processing inputs to work with scan op / function!?
        self.processed_input = process_batch_input_for_RNN(self._inputs)

        #initializing hidden state with a shape of [batch_size, hidden_nodes] 
        self.initial_hidden = self._inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([self.input_nodes, self.hidden_nodes]))

        #Function for gRU
        def gru (self, previous_hidden_state, x):
            
            #r and z dimension will be R * 1
            r = tf.sigmoid(tf.add(tf.matmul(self.W_r, x), tf.matmul(self.U_r, previous_hidden_state)))
            z = tf.sigmoid(tf.add(tf.matmul(self.W_z, x), tf.matmul(self.U_z, previous_hidden_state)))

            h_ = tf.tanh(tf.add(tf.matmul(self.W_h, x), tf.matmul(self.U_h, tf.matmul(r, previous_hidden_state))))

            current_hidden_state = tf.add(tf.multiply((1-z), h_), tf.multiply(z, previous_hidden_state))