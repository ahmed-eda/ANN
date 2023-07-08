print('test')

import tensorflow as tf
# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
#sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
#print(sess)
#print(config=tf.ConfigProto(device_count={'GPU': 1}))
print('ending')