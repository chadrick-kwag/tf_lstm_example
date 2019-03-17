"""
first rnn example

this one will only use the last output of the rnn output.
also the output of the rnn will go through one fc layer with linear activation.
"""


import tensorflow as tf 
import numpy as np 
import os, sys, random, shutil


ckpt_dir = "ckpt_t1"

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)

test_sequence=[
    1,2,1,2,1,5,1,3,1,2,1,3,1
]


batch_size=1

sequence_length=4
answer_length=1
crop_length = sequence_length + answer_length

sequence_num = len(test_sequence) - (crop_length-1)

train_data=[]

for i in range(sequence_num):
    crop = test_sequence[i:i+crop_length]

    train_sequence = crop[:sequence_length]
    label_sequence = crop[sequence_length:]

    train_data.append([train_sequence, label_sequence])
    


print(train_data)


# sys.exit(0)

print("building model")



input_ph = tf.placeholder(tf.float32, shape=(None,sequence_length,1)) # [batch_size, sequence length, input dimension]
label_ph = tf.placeholder(tf.float32, shape=(None, answer_length,1))

# label = tf.expand_dims(label_ph, axis=1)

# print("label ts:{}".format(label))

lstm_layer = tf.keras.layers.LSTM(10)

lstm_output = lstm_layer(input_ph)

fc_output = tf.contrib.layers.fully_connected(lstm_output,1, activation_fn=None)

fc_output_expanded = tf.expand_dims(fc_output, axis=1)



# last_output = output[:,-1]
# last_output = tf.reshape(last_output, [-1, 1])
# last_output = tf.expand_dims(last_output, axis=1)

print("fc_output ts: {}".format(fc_output))


loss_ts = tf.losses.mean_squared_error(label_ph, fc_output_expanded)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

opt_op = optimizer.minimize(loss_ts)



# print("output tensor: {}".format(output))

writer = tf.summary.FileWriter("tfsummary")
saver = tf.train.Saver()




total_steps = 10000

with tf.Session() as sess:
    writer.add_graph(sess.graph)


    sess.run(tf.global_variables_initializer())

    for step in range(total_steps):

        # random select 
        pick = random.choices(train_data, k=batch_size)

        input_data=[]
        label_data=[]
        for data in pick:
            input_data.append(data[0])
            label_data.append(data[1])
        
        input_data = np.array(input_data)
        label_data = np.array(label_data)

        input_data = np.expand_dims(input_data, axis=2)
        label_data = np.expand_dims(label_data, axis=2)

        print("input_data shape:{}".format(input_data.shape))
        print("label_data shape:{}".format(label_data.shape))




        loss, _, pred_output = sess.run([loss_ts, opt_op, fc_output], feed_dict={
            input_ph : input_data,
            label_ph : label_data
        })

        print("loss={} , pred_output={}, true_value={}".format(loss, pred_output, label_data))

    save_prefix = os.path.join(ckpt_dir,"save")
    saver.save(sess, save_prefix)


print("====done====")

