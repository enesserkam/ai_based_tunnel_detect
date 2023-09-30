from elasticsearch import Elasticsearch
import tensorflow as tf
import pickle
from keras_preprocessing.sequence import pad_sequences
import tldextract


def preprocess_and_pad_sequences(texts, tokenizer):

    sequences = tokenizer.texts_to_sequences(texts)
    sequences = [url[-63:] if len(url) > 63 else url for url in sequences]
    padded_sequences = pad_sequences(sequences, maxlen=63)

    return padded_sequences


def calculate_mean(int_list):
    if len(int_list) != 5:
        raise ValueError("The input list must have exactly 5 elements.")

    total_sum = sum(int_list)
    mean = total_sum / len(int_list)
    return mean


def extract_subdomains(fqdn_list):
    subdomain_lists = []

    for fqdn in fqdn_list:
        ext = tldextract.extract(fqdn)
        subdomain = ext.subdomain
        subdomain_lists.append(subdomain)

    return subdomain_lists


with open("tokenizer.pkl", "rb") as f:
    tokenizer_ = pickle.load(f)

model = tf.keras.models.load_model("tunnel_detection_model.h5")

es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
index_name = 'test_dataset'
field_to_read = 'fqdn_samples'

scroll_query = {
    "size": 5000,
    "query": {
        "match_all": {}
    }
}

response = es.search(index=index_name, scroll="5m", body=scroll_query)
scroll_id = response["_scroll_id"]

while True:
    documents = response['hits']['hits']

    for document in documents:
        field_value = document['_source'][field_to_read]

        subdomain_list = extract_subdomains(field_value)
        padded_list = preprocess_and_pad_sequences(subdomain_list, tokenizer_)
        predictions = model.predict(padded_list)
        prediction = calculate_mean(predictions)

        mean_predictions = [calculate_mean(column_predictions) for column_predictions in predictions.T]

        doc_id = document['_id']
        update_query = {
            "doc": {
                "predicted_values": mean_predictions[0]
            }
        }
        es.update(index=index_name, id=doc_id, body=update_query)

    response = es.scroll(scroll_id=scroll_id, scroll="5m")

    if not response['hits']['hits']:
        break
