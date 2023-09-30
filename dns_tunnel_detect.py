import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D
from keras import regularizers
import mlflow
import mlflow.keras
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.metrics import Precision, Recall
import os
import joblib


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

        print(len(self.dns_tunnel_data))
        print(len(self.not_dns_tunnel_data))

        for x in self.dns_tunnel_data:
            temp_str = x[0]
            temp_str = temp_str.replace('-', '').strip()
            x[0] = temp_str

        for z in self.not_dns_tunnel_data:
            temp_str = z[0]
            temp_str = temp_str.replace('-', '').strip()
            z[0] = temp_str

    def split_data(self):

        dns_train, dns_val_test = train_test_split(self.dns_tunnel_data, test_size=0.2, random_state=42)
        not_dns_train, not_dns_val_test = train_test_split(self.not_dns_tunnel_data, test_size=0.2, random_state=42)

        dns_val, dns_test = train_test_split(dns_val_test, test_size=0.5, random_state=42)
        not_dns_val, not_dns_test = train_test_split(not_dns_val_test, test_size=0.5, random_state=42)

        return dns_train, dns_val, dns_test, not_dns_train, not_dns_val, not_dns_test

    def prepare_data(self, dns_train, dns_val, dns_test, not_dns_train, not_dns_val, not_dns_test):
        train_data = dns_train + not_dns_train
        val_data = dns_val + not_dns_val
        test_data = dns_test + not_dns_test

        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        # scaling for experimental use
        """
        train_data = random.sample(train_data, int(len(train_data) * scale))
        val_data = random.sample(val_data, int(len(val_data) * scale))
        """

        train_urls, train_labels = zip(*train_data)
        val_urls, val_labels = zip(*val_data)
        test_urls, test_labels = zip(*test_data)

        return train_urls, train_labels, val_urls, val_labels, test_urls, test_labels


# A function for creating a confusion matrix in order to upload to MlFlow as an artifact
def plot_confusion(normalized_conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(normalized_conf_matrix, annot=True, fmt=".4f", cmap="Blues",
                xticklabels=['Not Tunnel', 'Tunnel'],
                yticklabels=['Not Tunnel', 'Tunnel'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")

    if os.path.exists('Artifacts/conf.png'):
        os.remove('Artifacts/conf.png')

    plt.savefig('Artifacts/conf.png')
    mlflow.log_artifact('Artifacts/conf.png')

    plt.show()


# A function for creating an artifact plot in order to upload to MlFlow experiments
def plot_data(loss_list, acc_list, pre_list, rec_list):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    x = [1, 2, 3, 4, 5]

    axes[0, 0].plot(x, loss_list)
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel("Epoch")
    # Plot Accuracy
    axes[0, 1].plot(x, acc_list)
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel("Epoch")
    # Plot Precision
    axes[1, 0].plot(x, pre_list)
    axes[1, 0].set_title('Precision')
    axes[1, 0].set_xlabel("Epoch")
    # Plot Recall
    axes[1, 1].plot(x, rec_list)
    axes[1, 1].set_title('Recall')
    axes[1, 1].set_xlabel("Epoch")

    for ax in axes.flat:
        ax.set_xticks(x)

    plt.tight_layout()

    if os.path.exists('Artifacts/output.png'):
        os.remove('Artifacts/output.png')

    plt.savefig('Artifacts/output.png')
    mlflow.log_artifact('Artifacts/output.png')

    plt.show()


class TrainModel:
    def __init__(self):
        self.train_padded = []
        self.val_padded = []
        self.test_padded = []
        self.train_labels = []
        self.val_labels = []
        self.test_labels = []

# process_data function is to be sure that each token vector have same length
    def process_data(self, train_urls, train_labels, val_urls, val_labels, test_urls, test_labels):
        tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        tokenizer.fit_on_texts(train_urls)

        train_urls = [url[-63:] if len(url) > 63 else url for url in train_urls]
        val_urls = [url[-63:] if len(url) > 63 else url for url in val_urls]
        test_urls = [url[-63:] if len(url) > 63 else url for url in test_urls]

        train_sequences = tokenizer.texts_to_sequences(train_urls)
        val_sequences = tokenizer.texts_to_sequences(val_urls)
        test_sequences = tokenizer.texts_to_sequences(test_urls)

        self.train_padded = pad_sequences(train_sequences, maxlen=63)
        self.val_padded = pad_sequences(val_sequences, maxlen=63)
        self.test_padded = pad_sequences(test_sequences, maxlen=63)

        self.train_labels = np.array(train_labels, dtype=np.int32)
        self.val_labels = np.array(val_labels, dtype=np.int32)
        self.test_labels = np.array(test_labels, dtype=np.int32)

        return tokenizer

    def train_rnn_model(self, tokenizer):
        mlflow.set_experiment("v5")
        run_name = "Confusion_Matrix"
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("v5").experiment_id, run_name=run_name):

            model = Sequential()
            model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=63))

            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))

            model.add(LSTM(64, kernel_regularizer=regularizers.l2(0.01)))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            metrics = ['accuracy', Precision(), Recall()]
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

            history = model.fit(self.train_padded, self.train_labels, epochs=5, batch_size=256,
                                validation_data=(self.val_padded, self.val_labels))

            test_pred = model.predict(self.test_padded)

            test_pred_labels = (test_pred > 0.5).astype(int)

            test_results = model.evaluate(self.test_padded, self.test_labels, verbose=0)
            test_loss = test_results[0]
            test_acc = test_results[1]
            test_precision = precision_score(self.test_labels, test_pred_labels)
            test_recall = recall_score(self.test_labels, test_pred_labels)
                        
            print('Test Loss:', test_loss)
            print('Test Accuracy:', test_acc)
            print('Test Precision:', test_precision)
            print('Test Recall:', test_recall)

            mlflow.end_run()

            return model


processor = DataProcessor('datasets/tunnel_dataset_fixed.txt')
processor.read_data()

dns_train_, dns_val_, dns_test_, not_dns_train_, not_dns_val_, not_dns_test_ = processor.split_data()

url_train, label_train, url_val, label_val, url_test, label_test = \
    processor.prepare_data(dns_train_, dns_val_, dns_test_, not_dns_train_, not_dns_val_, not_dns_test_)

model_ = TrainModel()
tokenizer_ = model_.process_data(url_train, label_train, url_val, label_val, url_test, label_test)
rnn_model = model_.train_rnn_model(tokenizer_)

rnn_model.save("rnn_model.h5")
