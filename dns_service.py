from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D
from keras import regularizers
import mlflow
import mlflow.keras
from keras.metrics import Precision, Recall
import pickle

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


def max_length_of_elements(input_list):
    if not input_list:
        return 0

    max_length = len(input_list[0][0])
    for element in input_list:
        length = len(element[0])
        if length > max_length:
            max_length = length

    return max_length


def read_test_data(filename):
    dns_tunnel_data = []
    not_dns_tunnel_data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('\t')
            subdomain = line[0]
            label = line[1]
            if label == 'dns_tunnel':
                temp = [subdomain, 1]
                dns_tunnel_data.append(temp)
            else:
                temp = [subdomain, 0]
                not_dns_tunnel_data.append(temp)

    return dns_tunnel_data, not_dns_tunnel_data


class DataProcessor:
    def __init__(self, filename):
        self.filename = filename
        self.dns_tunnel_data = []
        self.not_dns_tunnel_data = []

    def read_data(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')

                if len(line) > 1:
                    subdomain = line[0]
                    label = line[1]

                    if label == 'dns_tunnel':
                        temp = [subdomain, 1]
                        self.dns_tunnel_data.append(temp)
                    else:
                        temp = [subdomain, 0]
                        self.not_dns_tunnel_data.append(temp)

        a = 0
        for x in self.dns_tunnel_data:
            temp_str = x[0]
            if a % 3 == 0:
                temp_str = temp_str.replace('-', '').strip()
            x[0] = temp_str
            a += 1

    def split_data(self):
        dns_train, dns_val = train_test_split(self.dns_tunnel_data, test_size=0.1, random_state=42)
        not_dns_train, not_dns_val = train_test_split(self.not_dns_tunnel_data, test_size=0.1, random_state=42)

        return dns_train, dns_val, not_dns_train, not_dns_val

    def prepare_data(self, dns_train, dns_val, not_dns_train, not_dns_val):
        scale = 0.33

        not_train_len = int(len(not_dns_train) * scale)
        not_val_len = int(len(not_dns_val) * scale)

        not_dns_train = not_dns_train[0:not_train_len + 1]
        not_dns_val = not_dns_val[0:not_val_len + 1]

        train_data = dns_train + not_dns_train
        val_data = dns_val + not_dns_val

        random.shuffle(train_data)
        random.shuffle(val_data)

        # scaling for experimental use

        """
        train_data = random.sample(train_data, int(len(train_data) * scale))
        val_data = random.sample(val_data, int(len(val_data) * scale))
        """

        train_urls, train_labels = zip(*train_data)
        val_urls, val_labels = zip(*val_data)

        return train_urls, train_labels, val_urls, val_labels


class TrainModel:
    def __init__(self):
        self.train_padded = []
        self.val_padded = []
        self.test_padded = []
        self.train_labels = []
        self.val_labels = []
        self.test_labels = []

    def process_data(self, train_urls, train_labels, val_urls, val_labels):
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(train_urls)

        def save_tokenizer(token, filename):
            with open(filename, 'wb') as f:
                pickle.dump(token, f)

        save_tokenizer(tokenizer, 'elastic_write/tokenizer.pkl')

        train_urls = [url[-63:] if len(url) > 63 else url for url in train_urls]
        val_urls = [url[-63:] if len(url) > 63 else url for url in val_urls]

        train_sequences = tokenizer.texts_to_sequences(train_urls)
        val_sequences = tokenizer.texts_to_sequences(val_urls)

        self.train_padded = pad_sequences(train_sequences, maxlen=63)
        self.val_padded = pad_sequences(val_sequences, maxlen=63)

        self.train_labels = np.array(train_labels, dtype=np.int32)
        self.val_labels = np.array(val_labels, dtype=np.int32)

        return tokenizer

    def train_rnn_model(self, tokenizer):

        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=63))

        model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        metrics = ['accuracy', Precision(), Recall()]
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

        mlflow.log_param("embedding_dim", 100)
        mlflow.log_param("conv_filters", 64)
        mlflow.log_param("conv_kernel_size", 3)
        mlflow.log_param("lstm_units", 64)
        mlflow.log_param("dense_units", 32)
        mlflow.log_param("l2_regularizer", 0.01)
        mlflow.log_param("batch_size", 256)
        mlflow.log_param("epochs", 3)

        history = model.fit(self.train_padded, self.train_labels, epochs=5, batch_size=256,
                            validation_data=(self.val_padded, self.val_labels))

        model.save("tunnel_detection_model.h5")

        return model


processor = DataProcessor('datasets/tunnel_dataset_fixed.txt')
processor.read_data()

dns_train_, dns_val_, not_dns_train_, not_dns_val_ = processor.split_data()

url_train, label_train, url_val, label_val = processor.prepare_data(dns_train_, dns_val_, not_dns_train_, not_dns_val_)

model_ = TrainModel()
tokenizer_ = model_.process_data(url_train, label_train, url_val, label_val)
rnn_model = model_.train_rnn_model(tokenizer_)

