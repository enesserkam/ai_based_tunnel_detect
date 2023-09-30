from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json

host = 'localhost'
port = 9200
scheme = 'http'

es = Elasticsearch([{'host': host, 'port': port, 'scheme': scheme}])

file_path = "erkam_dataset.json"

with open(file_path, 'r') as json_file:
    json_data = json.load(json_file)

actions = [
    {
        "_index": "ml_dataset",
        "_source": document
    }
    for document in json_data
]

success, _ = bulk(es, actions, index='_index')
print(f"Successfully indexed {success} documents")
