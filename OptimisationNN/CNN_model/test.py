import numpy as np
import pickle
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Activation, Reshape, Layer
from keras.models import Model, load_model, model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# GPU memory configuration
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Character set used for decoding predictions
chars = ["A", "B", "C", "D", "E", "H", "K", "M", "O", "P", "T", "X", "Y",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Custom normalization layer used in model
class NormLayer(Layer):
    def __init__(self, **kwargs):
        super(NormLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernal = self.add_weight(
            name='NormLayer',
            shape=(1, 15),
            initializer='ones',
            trainable=True
        )
        super(NormLayer, self).build(input_shape)

    def call(self, inputs):
        out = K.dot(self.kernal, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        return out[:, 0, :]
   
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 23)

    def get_config(self):
        base_config = super(NormLayer, self).get_config()
        return dict(list(base_config.items()))

# Load trained model with custom layer
e2e_model = load_model('e2e_model.h5', custom_objects={'NormLayer': NormLayer})

# Load label data
label_path = 'label.txt'
tem_label = np.loadtxt(label_path)
row, col = tem_label.shape

# Prepare one-hot encoded labels
label1 = np.zeros([row, 13])
label2 = np.zeros([row, 10])
label3 = np.zeros([row, 10])
label4 = np.zeros([row, 10])
label5 = np.zeros([row, 10])
label6 = np.zeros([row, 13])
label7 = np.zeros([row, 10])
label8 = np.zeros([row, 10])

for i in range(row):
    label1[i, int(tem_label[i, 0])] = 1
    label2[i, int(tem_label[i, 1]) - 13] = 1
    label3[i, int(tem_label[i, 2]) - 13] = 1
    label4[i, int(tem_label[i, 3]) - 13] = 1
    label5[i, int(tem_label[i, 4])] = 1
    label6[i, int(tem_label[i, 5])] = 1
    label7[i, int(tem_label[i, 6]) - 13] = 1
    label8[i, int(tem_label[i, 7]) - 13] = 1

# Load image data
img_path = './train_data.pkl'
img_data = np.load(img_path, allow_pickle=True)
img_data = img_data.transpose(0, 2, 1, 3)

# Run predictions on first 10 samples
e2e_predict = e2e_model.predict(img_data[0:10, :, :, :])

# Print true label for a given sample
def print_trueLabel(num):
    print(np.array([
        np.argmax(label1[num, :]), np.argmax(label2[num, :]) + 13,
        np.argmax(label3[num, :]) + 13, np.argmax(label4[num, :]) + 13,
        np.argmax(label5[num, :]), np.argmax(label6[num, :]),
        np.argmax(label7[num, :]) + 13, np.argmax(label8[num, :]) + 13
    ]))
    print(
        chars[np.argmax(label1[num, :])] +
        chars[np.argmax(label2[num, :]) + 13] +
        chars[np.argmax(label3[num, :]) + 13] +
        chars[np.argmax(label4[num, :]) + 13] +
        chars[np.argmax(label5[num, :])] +
        chars[np.argmax(label6[num, :])] +
        chars[np.argmax(label7[num, :]) + 13] +
        chars[np.argmax(label8[num, :]) + 13]
    )

# Print predicted label for a batch of samples
def print_predictLabel(x):
    num = x[0].shape[0]
    sort = np.zeros([num, len(x)])
    for i in range(len(x)):
        temp = x[i]
        sort[:, i] = np.argmax(temp, axis=1)
    for i in range(num):
        print(np.array([
            int(sort[i, 0]), int(sort[i, 1]) + 13, int(sort[i, 2]) + 13,
            int(sort[i, 3]) + 13, int(sort[i, 4]), int(sort[i, 5]),
            int(sort[i, 6]) + 13, int(sort[i, 7]) + 13
        ]))
        print(
            chars[int(sort[i, 0])] +
            chars[int(sort[i, 1]) + 13] +
            chars[int(sort[i, 2]) + 13] +
            chars[int(sort[i, 3]) + 13] +
            chars[int(sort[i, 4])] +
            chars[int(sort[i, 5])] +
            chars[int(sort[i, 6]) + 13] +
            chars[int(sort[i, 7]) + 13]
        )

# Print true labels for first 10 samples
print_trueLabel(0)
print_trueLabel(1)
print_trueLabel(2)
print_trueLabel(3)
print_trueLabel(4)
print_trueLabel(5)
print_trueLabel(6)
print_trueLabel(7)
print_trueLabel(8)
print_trueLabel(9)

# Print predicted labels for same samples
print_predictLabel(e2e_predict)