# Note to my myself

Below is another manner to implement a fully connected network,
this time in the case of a regression.

Because it is a regression, the loss here is obviously different
from the one for the classification of the Iris dataset.

~~~~
with tf.name_scope("dnn"):    
    
    initializer_w = tf.contrib.layers.xavier_initializer()
    initializer_b = tf.zeros_initializer()    
    
    w1 = tf.Variable(initializer_w(shape=[nb_features, hidden_layer_nodes]), name="w1")
    b1 = tf.Variable(initializer_b(shape=[hidden_layer_nodes]), name="b1")
    z1 = tf.add(tf.matmul(X_data, w1), b1, name="z1")
    a1 = tf.nn.leaky_relu(z1, name="a1")
    
    w2 = tf.Variable(initializer_w(shape=[hidden_layer_nodes, hidden_layer_nodes]), name="w2")
    b2 = tf.Variable(initializer_b(shape=[hidden_layer_nodes]), name="b2")
    z2 = tf.add(tf.matmul(a1, w2), b2, name="z2")
    a2 = tf.nn.tanh(z2, name="a2")
    drop_out_2 = tf.layers.dropout(a2, dropout_rate, training, name="dropout_2")   
    
    w3 = tf.Variable(initializer_w(shape=[hidden_layer_nodes, nb_output_neurons]), name="w3")
    b3 = tf.Variable(initializer_b(shape=[nb_output_neurons]), name="b3")
    # no activation for the final layer in regression
    final_output = tf.add(tf.matmul(drop_out_2, w3), b3, name="final_output")
  
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.pow(final_output - y_target, 2), name="loss")
  
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
~~~~