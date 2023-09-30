import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.keras
from elasticsearch.helpers import bulk

host = 'localhost'
port = 9200
scheme = 'http'
es_ = Elasticsearch([{'host': host, 'port': port, 'scheme': scheme}])


# uploads the tested data to elasticsearch as a new index in order to examine the experimental results
def upload_to_elasticsearch(dataf, es):
    actions = [
        {
            "_op_type": "index",
            "_index": "prediction_test_modified",
            "_source": row.to_dict()
        }
        for _, row in dataf.iterrows()
    ]
    success, failed = bulk(es, actions)
    print(f"Successfully inserted: {success} | Failed: {failed}")


class EnsembleModel:
    def __init__(self, data_file):
        self.data_file = data_file

    def map_to_label(self, predictions, mapping):
        return [mapping[pred] for pred in predictions]

    def read_test(self):
        df = pd.read_csv(self.data_file)
        return df

    def custom_conversion(self, value):
        if value == "None":
            return -1
        try:
            return float(value)
        except:
            return value

    def label_conversion(self, value):
        if value == "suspicious":
            return 0.5
        elif value == "pass":
            return 0
        elif value == "low":
            return 0
        elif value == "high":
            return 1
        try:
            return float(value)
        except:
            return value

    def fix_data(self):
        df = self.read_test()
        df.fillna(-1, inplace=True)
        categorical_columns = ['record_top1.type', 'record_top2.type']
        for column in categorical_columns:
            df[column] = pd.factorize(df[column])[0]

        columns_to_convert = ['other.fail_ratio', 'other.fqdn_count', 'other.fqdn_hit_ratio', 'other.hits']
        df[columns_to_convert] = df[columns_to_convert].applymap(self.custom_conversion)
        # df["is_tunnel"] = df["is_tunnel"].apply(self.label_conversion)
        label_mapping = {'pass': 0, 'suspicious': 1, 'high': 2}
        df['is_tunnel'] = df['is_tunnel'].map(label_mapping)
        n_rows, n_cols = df.shape

        print(f"ROWS: {n_rows} || COLS: {n_cols}")

        return df

    def train_upper_model(self, df):

        mlflow.set_experiment("Ensemble_test")
        run_name = "deneme"
        with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name("Ensemble_test").experiment_id, run_name=run_name):
            rf_model = joblib.load("joblib/random_forest.joblib")
            xgb_model = joblib.load("joblib/xgb_model.joblib")
            lgb_model = joblib.load("joblib/lgb_model.joblib")

            mlflow.log_params({
                "xgb_n_estimators": xgb_model.get_params()["n_estimators"],
                "xgb_learning_rate": xgb_model.get_params()["learning_rate"],
                "xgb_nthread": xgb_model.get_params()["nthread"],
                "xgb_subsample": xgb_model.get_params()["subsample"],
                "xgb_min_child_weight": xgb_model.get_params()["min_child_weight"],
                "xgb_max_depth": xgb_model.get_params()["max_depth"],

                "lgb_n_estimators": lgb_model.get_params()["n_estimators"],
                "lgb_max_depth": lgb_model.get_params()["max_depth"],
                "lgb_min_child_samples": lgb_model.get_params()["min_child_samples"],
                "lgb_min_child_weight": lgb_model.get_params()["min_child_weight"],
                "lgb_boosting_type": lgb_model.get_params()["boosting_type"],
                "lgb_bagging_freq": lgb_model.get_params()["bagging_freq"],
                "lgb_num_leaves": lgb_model.get_params()["num_leaves"],
                "lgb_learning_rate": lgb_model.get_params()["learning_rate"],
                "lgb_subsample": lgb_model.get_params()["subsample"],
                "lgb_colsample_bytree": lgb_model.get_params()["colsample_bytree"],
                "lgb_reg_alpha": lgb_model.get_params()["reg_alpha"],
                "lgb_reg_lambda": lgb_model.get_params()["reg_lambda"],

                "rf_n_estimators": rf_model.get_params()["n_estimators"],
                "rf_max_depth": rf_model.get_params()["max_depth"],
                "rf_min_samples_split": rf_model.get_params()["min_samples_split"],
                "rf_min_samples_leaf": rf_model.get_params()["min_samples_leaf"],
                "rf_bootstrap": rf_model.get_params()["bootstrap"],
                "rf_class_weight": rf_model.get_params()["class_weight"],
                "rf_max_features": rf_model.get_params()["max_features"],
            })

            X = df.iloc[:, :-1]
            X = X.drop('fqdn_samples', axis=1)
            X = X.drop('domain', axis=1)
            X['predicted_values'] = X['predicted_values'].apply(lambda x: min(3 * x, 0.9))
            print(X.shape)
            y = df.iloc[:, -1]

            rf_pred = rf_model.predict(X)
            xgb_pred = xgb_model.predict(X)
            lgb_pred = lgb_model.predict(X)

            mean_pred = np.mean([xgb_pred, lgb_pred, rf_pred], axis=0)
            pred = np.divide(mean_pred, 2)
            pred_list = []
            for i in range(len(pred)):
                if pred[i] < 0:
                    print("Pred < 0")
                elif 0 <= pred[i] < 0.3:
                    pred_list.append(0)
                elif 0.3 <= pred[i] < 0.75:
                    pred_list.append(1)
                elif 0.75 <= pred[i] <= 1:
                    pred_list.append(2)
                else:
                    print("Pred > 1")

            print("Pred 0: ", pred_list.count(0))
            print("Pred 1: ", pred_list.count(1))
            print("Pred 2: ", pred_list.count(2))

            pred_array = np.array(pred_list)
            label_mapping = {'pass': 0, 'low': 0, 'suspicious': 1, 'high': 2}
            df['ensemble_predictions'] = pred_array
            mapping_dict = {0: 'pass', 1: 'suspicious', 2: 'high'}
            df['is_tunnel'] = df['is_tunnel'].map(mapping_dict)
            df['ensemble_predictions'] = df['ensemble_predictions'].map(mapping_dict)
            upload_to_elasticsearch(df, es_)

            accuracy_rf = accuracy_score(y, rf_pred)
            precision_rf = precision_score(y, rf_pred, average=None)
            if len(precision_rf) == 2:
                precision_rf = np.insert(precision_rf, 0, 0)
            recall_rf = recall_score(y, rf_pred, average=None)
            if len(recall_rf) == 2:
                recall_rf = np.insert(recall_rf, 0, 0)
            f1_rf = f1_score(y, rf_pred, average=None)
            if len(f1_rf) == 2:
                f1_rf = np.insert(f1_rf, 0, 0)
            print(f"---------------Random Forest------------------")
            for class_label in label_mapping.keys():
                if class_label in ['suspicious', 'high']:
                    class_idx = label_mapping[class_label]
                    print(f"Metrics for class '{class_label}':")
                    print(f"  Precision: {precision_rf[class_idx]:.4f}")
                    print(f"  Recall: {recall_rf[class_idx]:.4f}")
                    print(f"  F1-score: {f1_rf[class_idx]:.4f}")
                    print("--------------------")

                    mlflow.log_metric(f"RF Precision metric for {class_label}", precision_rf[class_idx])
                    mlflow.log_metric(f"RF Recall metric for {class_label}", recall_rf[class_idx])
                    mlflow.log_metric(f"RF F-1 score for {class_label}", f1_rf[class_idx])

            accuracy_lgbm = accuracy_score(y, lgb_pred)
            precision_lgbm = precision_score(y, lgb_pred, average=None)
            print(precision_lgbm)
            recall_lgbm = recall_score(y, lgb_pred, average=None)
            f1_lgbm = f1_score(y, lgb_pred, average=None)
            print(f"---------------Light GBM------------------")
            for class_label in label_mapping.keys():
                if class_label in ['suspicious', 'high']:
                    class_idx = label_mapping[class_label]
                    print(f"Metrics for class '{class_label}':")
                    print(f"  Precision: {precision_lgbm[class_idx]:.4f}")
                    print(f"  Recall: {recall_lgbm[class_idx]:.4f}")
                    print(f"  F1-score: {f1_lgbm[class_idx]:.4f}")
                    print("--------------------")

                    mlflow.log_metric(f"L_GBM Precision metric for {class_label}", precision_lgbm[class_idx])
                    mlflow.log_metric(f"L_GBM Recall metric for {class_label}", recall_lgbm[class_idx])
                    mlflow.log_metric(f"L_GBM F-1 score for {class_label}", f1_lgbm[class_idx])

            accuracy_xgb = accuracy_score(y, xgb_pred)
            precision_xgb = precision_score(y, xgb_pred, average=None)
            print(precision_xgb)
            recall_xgb = recall_score(y, xgb_pred, average=None)

            f1_xgb = f1_score(y, xgb_pred, average=None)
            print(f"---------------XGBoost------------------")
            for class_label in label_mapping.keys():
                if class_label in ['suspicious', 'high']:
                    class_idx = label_mapping[class_label]
                    print(f"Metrics for class '{class_label}':")
                    print(f"  Precision: {precision_xgb[class_idx]:.4f}")
                    print(f"  Recall: {recall_xgb[class_idx]:.4f}")
                    print(f"  F1-score: {f1_xgb[class_idx]:.4f}")
                    print("--------------------")
                    mlflow.log_metric(f"XGBoost Precision metric for {class_label}", precision_xgb[class_idx])
                    mlflow.log_metric(f"XGBoost Recall metric for {class_label}", recall_xgb[class_idx])
                    mlflow.log_metric(f"XGBoost F-1 score for {class_label}", f1_xgb[class_idx])

            precision_mean = precision_score(y, pred_array, average=None)
            if len(precision_mean) == 2:
                precision_mean = np.insert(precision_mean, 0, 0)
            recall_mean = recall_score(y, pred_array, average=None)
            if len(recall_mean) == 2:
                recall_mean = np.insert(recall_mean, 0, 0)
            f1_mean = f1_score(y, pred_array, average=None)
            if len(f1_mean) == 2:
                f1_mean = np.insert(f1_mean, 0, 0)
            print(f"---------------ENSEMBLE------------------")
            for class_label in label_mapping.keys():
                if class_label in ['suspicious', 'high']:
                    class_idx = label_mapping[class_label]
                    print(f"Metrics for class '{class_label}':")
                    print(f"  Precision: {precision_mean[class_idx]:.4f}")
                    print(f"  Recall: {recall_mean[class_idx]:.4f}")
                    print(f"  F1-score: {f1_mean[class_idx]:.4f}")
                    print("--------------------")

                    mlflow.log_metric(f"Ensemble Precision metric for {class_label}", precision_mean[class_idx])
                    mlflow.log_metric(f"Ensemble Recall metric for {class_label}", recall_mean[class_idx])
                    mlflow.log_metric(f"Ensemble F-1 score for {class_label}", f1_mean[class_idx])

            mlflow.end_run()


test_file = "test_exported_data.csv"
ensemble_model = EnsembleModel(test_file)
dataframe = ensemble_model.fix_data()

ensemble_model.train_upper_model(dataframe)
