import csv
from elasticsearch import Elasticsearch


class ImportData:
    def __init__(self, index_name, fields_to_export):
        self.index_name = index_name
        self.fields_to_export = fields_to_export

    def import_data(self):
        es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

        search_params = {
            "index": self.index_name,
            "size": 5000,
            "scroll": "15m",
            "body": {
                "query": {
                    "match_all": {}
                }
            }
        }

        initial_results = es.search(**search_params)
        scroll_id = initial_results['_scroll_id']

        csv_file_path = "../exported_data.csv"

        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(self.fields_to_export)

            while True:
                for hit in initial_results['hits']['hits']:
                    row_data = [self.get_nested_field(hit['_source'], field) for field in self.fields_to_export]
                    csv_writer.writerow(row_data)

                scroll_results = es.scroll(scroll_id=scroll_id, scroll="5m")

                if not scroll_results['hits']['hits']:
                    break

                scroll_id = scroll_results['_scroll_id']
                initial_results = scroll_results
        print("Export completed and saved as:", csv_file_path)

    def get_nested_field(self, source_dict, field):
        keys = field.split('.')
        value = source_dict
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, "")
            else:
                value = ""
                break
        return str(value)


idx_name = "ml_dataset"
fields = ["res_dga.dga", "res_dga.success", "res_dga.fail", "res_dga.ratio", "res_dga.not_dga", "company_lw.fqdn_hit_ratio", "company_lw.fqdn_count", "company_lw.fail_ratio", "company_lw.hits",
          "company.fail_ratio", "company.fail_count", "company.fqdn_count", "company.fqdn_hit_ratio", "company.hits",
          "company.query_stats.A.count", "company.query_stats.A.ratio", "company.query_stats.AAAA.count", "company.query_stats.AAAA.ratio",
          "company.query_stats.CNAME.count", "company.query_stats.CNAME.ratio", "company.query_stats.HTTPS.count", "company.query_stats.HTTPS.ratio",
          "company.query_stats.MX.count", "company.query_stats.MX.ratio", "company.query_stats.None.count", "company.query_stats.None.ratio",
          "company.query_stats.NS.count", "company.query_stats.NS.ratio", "company.query_stats.SOA.count", "company.query_stats.SOA.ratio",
          "company.query_stats.SPF.count", "company.query_stats.SPF.ratio", "company.query_stats.SRV.count", "company.query_stats.SRV.ratio",
          "company.query_stats.TXT.count", "company.query_stats.TXT.ratio", "company.success_count", "company.unique_response",
          "eliminated.other_comp.count", "eliminated.other_comp.ratio", "eliminated.q_type.count", "eliminated.q_type.ratio",
          "other.fail_ratio", "other.fqdn_count", "other.fqdn_hit_ratio", "other.hits", "other.unique_company",
          "record_top1.count", "record_top1.ratio", "record_top1.type", "record_top2.count", "record_top2.ratio", "record_top2.type", "predicted_values", "domain",
          "fqdn_samples", "is_tunnel"
          ]

data_importer = ImportData(idx_name, fields)
data_importer.import_data()
