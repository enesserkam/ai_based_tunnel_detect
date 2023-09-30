import time

import pandas as pd
from upper_models.Random_Forest import RandomForestModel
from upper_models.XG_Boost import XGBoostModel
from upper_models.Light_GBM import LightGBMModel


class DataFrame:
    def __init__(self, filename):
        self.filename = filename

# fills the empty data or returns float values
    def custom_conversion(self, value):
        if value == "None":
            return -1
        try:
            return float(value)
        except:
            return value

    def read_data(self):
        df = pd.read_csv(self.filename)
        return df

# giving numerical values to categorical columns
    def fix_data(self):
        df = self.read_data()
        df.fillna(-1, inplace=True)
        categorical_columns = ['record_top1.type', 'record_top2.type']
        for column in categorical_columns:
            df[column] = pd.factorize(df[column])[0]

        columns_to_convert = ['other.fail_ratio', 'other.fqdn_count', 'other.fqdn_hit_ratio', 'other.hits']
        df[columns_to_convert] = df[columns_to_convert].applymap(self.custom_conversion)
        df = df.drop('fqdn_samples', axis=1)
        df = df.drop('domain', axis=1)
        df['is_tunnel'] = df['is_tunnel'].replace("low", "pass")
        label_mapping = {'pass': 0, 'suspicious': 1, 'high': 2}
        df['is_tunnel'] = df['is_tunnel'].map(label_mapping)
        n_rows, n_cols = df.shape

        print(f"ROWS: {n_rows} || COLS: {n_cols}")

        return df

# in some cases, data may need to be fixed
    def split_data(self):
        df = self.fix_data()
        shuffled_df = df.sample(frac=1, random_state=19)

        split_index = int(0.9 * len(shuffled_df))

        df_train = shuffled_df.iloc[:split_index]
        df_test = shuffled_df.iloc[split_index:]

        return df_train, df_test


class TrainModels:
    def __init__(self, filename):
        self.filename = filename

    def train_models(self):
        DataF = DataFrame(self.filename)
        data_train = DataF.fix_data()

        start_time = time.time()

        print("---Training Random Forest---")
        Random_F = RandomForestModel(data_train)
        rf_model = Random_F.train_rf()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function took {elapsed_time:.4f} seconds to execute.")

        start_time = time.time()

        print("---Training XGBoost---")
        xg_boost = XGBoostModel(data_train)
        xgb_model = xg_boost.train_xgb()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function took {elapsed_time:.4f} seconds to execute.")

        start_time = time.time()

        print("---Training LightGBM---")
        light_gbm = LightGBMModel(data_train)
        lgb_model = light_gbm.train_lgb()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function took {elapsed_time:.4f} seconds to execute.")

        return data_train


file_name = "exported_data.csv"
train = TrainModels(file_name)
train.train_models()
