import tensorflow as tf , numpy as np 

# build model

test_sequence=[
    [1,2,1,2], # 1
    [2,1,2,1], # 5
    [2,1,5,1], # 3
    [6,7,4,5]
]

sequence_length=4
answer_length=1


input_ph = tf.placeholder(tf.float32, shape=(None,sequence_length,1)) # [batch_size, sequence length, input dimension]
label_ph = tf.placeholder(tf.float32, shape=(None, answer_length,1))

# label = tf.expand_dims(label_ph, axis=1)

# print("label ts:{}".format(label))

lstm_layer = tf.keras.layers.LSTM(10)

lstm_output = lstm_layer(input_ph)

fc_output = tf.contrib.layers.fully_connected(lstm_output,1, activation_fn=None)

fc_output_expanded = tf.expand_dims(fc_output, axis=1)



saver = tf.train.Saver()

ckpt_path = "ckpt_0/save"

with tf.Session() as sess:
    saver.restore(sess, ckpt_path)

    reshaped_test_sequence = np.expand_dims(test_sequence, axis=2)

    pred_output = sess.run(fc_output, feed_dict={
        input_ph: reshaped_test_sequence,

    })

    print("pred_output:{}".format(pred_output))

print("done")



