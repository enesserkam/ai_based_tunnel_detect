import argparse
import statistics
from elasticsearch import Elasticsearch, helpers
import tensorflow as tf
import pickle
from keras_preprocessing.sequence import pad_sequences
import tldextract


class PrepData:
    def preprocess_and_pad_sequences(self, texts, tokenizer):
        sequences = tokenizer.texts_to_sequences(texts)
        sequences = [url[-63:] if len(url) > 63 else url for url in sequences]
        padded_sequences = pad_sequences(sequences, maxlen=63)

        return padded_sequences

    def extract_subdomains(self, fqdn_list):
        subdomain_lists = []
        for fqdn in fqdn_list:
            ext = tldextract.extract(fqdn)
            subdomain = ext.subdomain
            subdomain_lists.append(subdomain)

        return subdomain_lists

    def get_predict_results(self, fqdns, model, tokenizer_):
        results = {}
        print(len(fqdns))
        subdomain_list = self.extract_subdomains(fqdns)
        padded_list = self.preprocess_and_pad_sequences(subdomain_list, tokenizer_)
        predictions = model.predict(padded_list)
        for i, pred in enumerate(predictions):
            results[fqdns[i]] = round(list(pred)[0], 5)
        print(len(results))
        return results


class Main:
    def main(self):
        with open("tokenizer.pkl", "rb") as f:
            tokenizer_ = pickle.load(f)

        model = tf.keras.models.load_model("tunnel_detection_model.h5")

        es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
        index_name = 'test_dataset'
        field_to_read = 'fqdn_samples'

        scroll_query = {
            "size": 10000,  # Number of documents per scroll batch
            "query": {
                "match_all": {}
            },
            "_source_includes": ["fqdn_samples"]
        }

        response = es.search(index=index_name, scroll="5m", body=scroll_query)
        scroll_id = response["_scroll_id"]

        documents_processed = 0

        while True:
            documents = response['hits']['hits']
            fqdns = []
            for sample in documents:
                fqdns += sample["_source"]["fqdn_samples"]
            prep = PrepData()
            predict_results = prep.get_predict_results(fqdns, model, tokenizer_)
            s_data = []
            for document in documents:
                field_value = document['_source'][field_to_read]
                predict_values = []

                for fqdn in field_value:
                    predict_values.append(predict_results[fqdn])

                doc_id = document['_id']
                s_data.append({"_op_type": "update", "_index": index_name, "_id": doc_id, "doc": {
                        "predicted_values": statistics.mean(predict_values)
                    }})

                documents_processed += 1
            helpers.bulk(es, s_data, raise_on_error=True)
            print(f"{len(s_data)} data processed")
            response = es.scroll(scroll_id=scroll_id, scroll="30m")

            if not response['hits']['hits']:
                break

    def argument_parsing(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-pr", "--process", required=False, help='process_count', type=int, default=5)
        args = parser.parse_args()

        return args


mainSample = Main()
args_ = mainSample.argument_parsing()
mainSample.main()