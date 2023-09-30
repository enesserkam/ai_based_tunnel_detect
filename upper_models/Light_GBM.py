from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib


class LightGBMModel:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_lgb(self):
        df = self.dataframe

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = y.astype('category')

        n_rows, n_cols = X.shape
        print(f"X || ROWS: {n_rows} || COLS: {n_cols}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        """
        lgbm_model = LGBMRegressor(n_estimators=100, random_state=42)
        lgbm_model.fit(X_train, y_train)
        y_pred = lgbm_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (LightGBM): {mse}\n")
        """

        lgbm_model = LGBMClassifier(
            n_estimators=1000,
            max_depth=20,
            min_child_samples=5,
            min_child_weight=0.001,
            boosting_type="gbdt",
            bagging_freq=5,
            num_leaves=128,
            learning_rate=0.001,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=-1,
            random_state=42
        )

        lgbm_model.fit(X_train, y_train)
        y_pred = lgbm_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (LightGBM): {accuracy}\n")
        joblib.dump(lgbm_model, "joblib/lgb_model.joblib")

        return lgbm_model
