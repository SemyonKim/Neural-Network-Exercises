import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Conv2D, MaxPooling2D,
    Activation, Reshape, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

# Force GPU device 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configure TensorFlow session to allow dynamic GPU memory growth
def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

configure_gpu()


# Custom normalization layer
class NormLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='NormLayer',
            shape=(1, input_shape[-1]),
            initializer='ones',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        out = K.dot(self.kernel, inputs)
        out = K.permute_dimensions(out, (1, 0, 2))
        return out[:, 0, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# Training class for end-to-end license plate recognition
class TrainE2E:
    def __init__(self):
        self.first_num = 13       # Number of classes for first character (letters)
        self.other_num = 10       # Number of classes for digits
        self.shape = (140, 30, 3) # Input image shape
        self.init_lr = 0.01       # Initial learning rate
        self.model = None

    def load_data(self):
        # Load labels
        label_path = './label.txt'
        self.tem_label = np.loadtxt(label_path)

        row, _ = self.tem_label.shape
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

        # Load images
        img_path = './train_data.pkl'
        self.img_data = np.load(img_path, allow_pickle=True)
        self.img_data = self.img_data.transpose(0, 2, 1, 3)

    def step_decay(self, epoch, lr):
        if epoch % 2 == 0 and epoch != 0:
            new_lr = lr * 0.5
            print(f"Learning rate changed to {new_lr}")
            return new_lr
        return lr

    def __build_network(self):
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
        outputs = [
            Activation('softmax', name=f'out{i+1}')(NormLayer()(branch))
            for i, branch in enumerate([
                first_subject, other_subject, first_subject, other_subject,
                first_subject, first_subject, other_subject, other_subject
            ])
        ]

        # Compile model
        rmsprop = RMSprop(learning_rate=self.init_lr)
        self.model = Model(input_img, outputs)
        self.model.compile(
            loss=['categorical_crossentropy'] * 8,
            optimizer=rmsprop,
            metrics=['accuracy'],
            loss_weights=[1] * 8
        )
        self.model.summary()

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
    trainer = TrainE2E()
    trainer.load_data()
    trainer.train()