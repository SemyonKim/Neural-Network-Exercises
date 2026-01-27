import numpy as np
import pickle
from keras.layers import Dense, Input, BatchNormalization, Conv2D, Flatten, MaxPooling2D, Activation, Reshape, Layer, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import os

# Use GPU device 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure TensorFlow session to allow dynamic GPU memory growth
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# Custom layer for normalization
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
	print(inputs.shape)
        out = K.dot(self.kernal, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        print(out.shape)
        return out[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        base_config = super(NormLayer, self).get_config()
        return dict(list(base_config.items()))

# Updated TensorFlow session config (for TF 2.x compatibility)
def get_session():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(config=config)

# Training class for end-to-end license plate recognition
class train_e2e:
    def __init__(self):
        self.first_num = 13       # Number of classes for first character (letters)
        self.other_num = 10       # Number of classes for digits
        self.shape = (140, 30, 3) # Input image shape
        self.init_lr = 0.01       # Initial learning rate

    def load_data(self):
        label_path = './label.txt'
        self.tem_label = np.loadtxt(label_path)

        row, col = self.tem_label.shape
        self.label1 = np.zeros([row, self.first_num])
        self.label2 = np.zeros([row, self.other_num])
        self.label3 = np.zeros([row, self.other_num])
        self.label4 = np.zeros([row, self.other_num])
        self.label5 = np.zeros([row, self.first_num])
        self.label6 = np.zeros([row, self.other_num])
        self.label7 = np.zeros([row, self.other_num])
        self.label8 = np.zeros([row, self.other_num])

        for i in range(row):
            self.label1[i, int(self.tem_label[i, 0])] = 1
            self.label2[i, int(self.tem_label[i, 1]) - 13] = 1
            self.label3[i, int(self.tem_label[i, 2]) - 13] = 1
            self.label4[i, int(self.tem_label[i, 3]) - 13] = 1
            self.label5[i, int(self.tem_label[i, 4])] = 1
            self.label6[i, int(self.tem_label[i, 5])] = 1
            self.label7[i, int(self.tem_label[i, 6]) - 13] = 1
            self.label8[i, int(self.tem_label[i, 7]) - 13] = 1

        img_path = './train_data.pkl'
        self.img_data = np.load(img_path, allow_pickle=True)
        self.img_data = self.img_data.transpose(0, 2, 1, 3)

    def step_decay(self, epoch):
        if epoch % 2 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
        return K.get_value(self.model.optimizer.lr)

    def __build_network(self):
        tf.compat.v1.keras.backend.set_session(get_session())

        input_img = Input(shape=self.shape)
        base_conv = 32
        x = input_img

        # 3 blocks: Conv → BN → ReLU → MaxPool
        for i in range(3):
            x = Conv2D(base_conv * (2 ** i), (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Final convolution layers
        x = Conv2D(256, (3, 3))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(1024, (1, 1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Output branches
        first_subject = Conv2D(self.first_num, (1, 1))(x)
        other_subject = Conv2D(self.other_num, (1, 1))(x)

        first_subject = Activation('softmax')(first_subject)
        other_subject = Activation('softmax')(other_subject)

        first_subject = Reshape((-1, self.first_num))(first_subject)
        other_subject = Reshape((-1, self.other_num))(other_subject)

        # NormLayer branches
        x1 = NormLayer()(first_subject)
        x2 = NormLayer()(other_subject)
        x3 = NormLayer()(first_subject)
        x4 = NormLayer()(other_subject)
        x5 = NormLayer()(first_subject)
        x6 = NormLayer()(first_subject)
        x7 = NormLayer()(other_subject)
        x8 = NormLayer()(other_subject)

        # Final outputs
        out1 = Activation('softmax', name='out1')(x1)
        out2 = Activation('softmax', name='out2')(x2)
        out3 = Activation('softmax', name='out3')(x3)
        out4 = Activation('softmax', name='out4')(x4)
        out5 = Activation('softmax', name='out5')(x5)
        out6 = Activation('softmax', name='out6')(x6)
        out7 = Activation('softmax', name='out7')(x7)
        out8 = Activation('softmax', name='out8')(x8)

        # Compile model
        rmsprop = RMSprop(lr=self.init_lr)
        self.model = Model(input_img, [out1, out2, out3, out4, out5, out6, out7, out8])
        self.model.compile(
            loss=['categorical_crossentropy'] * 8,
            optimizer=rmsprop,
            metrics=['accuracy'],
            loss_weights=[1] * 8
        )
        print(self.model.summary())

    def train(self):
        self.__build_network()
        lrate = LearningRateScheduler(self.step_decay)
        self.model.fit(
            self.img_data,
            [self.label1, self.label2, self.label3, self.label4,
             self.label5, self.label6, self.label7, self.label8],
            epochs=20,
            batch_size=256,
            verbose=1,
            callbacks=[lrate],
            validation_split=0.1
        )
        self.model.save("e2e_model.h5")

if __name__ == "__main__":
    my_train_nn = train_e2e()
    my_train_nn.load_data()
    my_train_nn.train()
