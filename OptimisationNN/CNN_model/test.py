import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# -------------------------------
# GPU memory configuration
# -------------------------------
def configure_gpu(memory_fraction=0.4):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(memory_fraction * 1024 * 4)  # approx MB
                    )]
                )
        except RuntimeError as e:
            print(e)

configure_gpu()

# -------------------------------
# Character set used for decoding predictions
# -------------------------------
chars = [
    "A", "B", "C", "D", "E", "H", "K", "M", "O", "P", "T", "X", "Y",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

# -------------------------------
# Custom normalization layer
# -------------------------------
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
        return (input_shape[0], input_shape[-1])

# -------------------------------
# Load trained model with custom layer
# -------------------------------
e2e_model = load_model('e2e_model.h5', custom_objects={'NormLayer': NormLayer})

# -------------------------------
# Load label data
# -------------------------------
label_path = 'label.txt'
tem_label = np.loadtxt(label_path)
row, _ = tem_label.shape

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

# -------------------------------
# Load image data
# -------------------------------
img_path = './train_data.pkl'
img_data = np.load(img_path, allow_pickle=True)
img_data = img_data.transpose(0, 2, 1, 3)

# -------------------------------
# Run predictions on first 10 samples
# -------------------------------
e2e_predict = e2e_model.predict(img_data[:10])

# -------------------------------
# Utility functions
# -------------------------------
def decode_true_label(num):
    """Decode and print the true label for a given sample index."""
    indices = [
        np.argmax(label1[num, :]),
        np.argmax(label2[num, :]) + 13,
        np.argmax(label3[num, :]) + 13,
        np.argmax(label4[num, :]) + 13,
        np.argmax(label5[num, :]),
        np.argmax(label6[num, :]),
        np.argmax(label7[num, :]) + 13,
        np.argmax(label8[num, :]) + 13
    ]
    print(indices)
    print("".join(chars[idx] for idx in indices))


def decode_predicted_labels(predictions):
    """Decode and print predicted labels for a batch of samples."""
    num_samples = predictions[0].shape[0]
    decoded = np.zeros([num_samples, len(predictions)], dtype=int)

    for i, pred in enumerate(predictions):
        decoded[:, i] = np.argmax(pred, axis=1)

    for i in range(num_samples):
        indices = [
            int(decoded[i, 0]),
            int(decoded[i, 1]) + 13,
            int(decoded[i, 2]) + 13,
            int(decoded[i, 3]) + 13,
            int(decoded[i, 4]),
            int(decoded[i, 5]),
            int(decoded[i, 6]) + 13,
            int(decoded[i, 7]) + 13
        ]
        print(indices)
        print("".join(chars[idx] for idx in indices))

# -------------------------------
# Print true and predicted labels for first 10 samples
# -------------------------------
for i in range(10):
    decode_true_label(i)

decode_predicted_labels(e2e_predict)