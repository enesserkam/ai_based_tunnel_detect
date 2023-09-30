from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
from xgboost import XGBRegressor, XGBClassifier


class XGBoostModel:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_xgb(self):
        df = self.dataframe

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = y.astype('category')

        n_rows, n_cols = X.shape
        print(f"X || ROWS: {n_rows} || COLS: {n_cols}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (XGBoost): {mse}\n")
        """
        xgb_model = XGBClassifier(n_estimators=1000,
                                  learning_rate=0.001,
                                  nthread=-1,
                                  subsample=0.8,
                                  min_child_weight=15,
                                  max_depth=6,
                                  random_state=42)

        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (XGBoost): {accuracy}\n")

        joblib.dump(xgb_model, "joblib/xgb_model.joblib")

        return xgb_model
