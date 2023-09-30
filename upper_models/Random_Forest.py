from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib


class RandomForestModel:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_rf(self):
        df = self.dataframe

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y = y.astype('category')

        n_rows, n_cols = X.shape
        print(f"X || ROWS: {n_rows} || COLS: {n_cols}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestClassifier(n_estimators=1000,
                                          max_depth=20,
                                          min_samples_split=8,
                                          min_samples_leaf=1,
                                          bootstrap=True,
                                          n_jobs=-1,
                                          max_features="sqrt",
                                          random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (Random Forest): {accuracy}\n")

        joblib.dump(rf_model, "joblib/random_forest.joblib")

        return rf_model
