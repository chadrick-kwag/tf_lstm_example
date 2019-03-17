"""
second rnn example

this one will use all output from the rnn
the rnn output will go through fc layer with linear activation
"""


import tensorflow as tf 
import numpy as np 
import os, sys, random, shutil


ckpt_dir = "ckpt_t1"

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(ckpt_dir)

# train_data_input_sequences=[
#     [0,1,0,1,0,0,1],
#     [1,1,0,0,1,0,1,0,0,0,1]
# ]

# train_data_output_sequences=[
#     [0,0,1,0,1,0,1],
#     [0,0,1,0,0,0,0,0,1,0,0]
# ]


train_data=[
    [1,0,1,1,1,3],
    # [2,6,6,3,2,4,1]
]

input_data=[]
label_data=[]
for data in train_data:
    input_data.append(data[:-1])
    label_data.append(data[-1])

input_data = np.array(input_data)
label_data = np.array(label_data)

input_data = np.expand_dims(input_data, axis=2)
label_data = np.expand_dims(label_data, axis=2)

print("input_data shape:{}".format(input_data.shape))
print("label_data shape:{}".format(label_data.shape))


print("building model")

# model = tf.keras.Sequential()

input = tf.keras.Input(shape=(None,1))

lstm_output =tf.keras.layers.LSTM(512)(input)

pred_output = tf.keras.layers.Dense(1)(lstm_output)

model = tf.keras.Model(inputs=input, outputs=pred_output)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mean_squared_error" )

model.summary()




# # print("output tensor: {}".format(output))

# writer = tf.summary.FileWriter("tfsummary")
# saver = tf.train.Saver()
checkpoint_path = "keras_ckpt/save"

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                 save_weights_only=True,
                                                 verbose=1)



total_steps = 1



# for step in range(total_steps):

#     # random select 
#     pick = random.choices(train_data, k=1)

#     input_data=[]
#     label_data=[]
#     for data in pick:
#         input_data.append(data[:-1])
#         label_data.append(data[-1])
    
#     input_data = np.array(input_data)
#     label_data = np.array(label_data)

#     input_data = np.expand_dims(input_data, axis=2)
#     label_data = np.expand_dims(label_data, axis=2)

#     print("input_data shape:{}".format(input_data.shape))
#     print("label_data shape:{}".format(label_data.shape))




    # loss, _, pred_output = sess.run([loss_ts, opt_op, fc_output], feed_dict={
    #     input_ph : input_data,
    #     label_ph : label_data
    # })

model.fit(x=input_data, y=label_data, callbacks = [cp_callback])

        # print("loss={} , pred_output={}, true_value={}".format(loss, pred_output, label_data))

    # save_prefix = os.path.join(ckpt_dir,"save")
    # saver.save(sess, save_prefix)


print("====done====")

