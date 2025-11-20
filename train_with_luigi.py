import mlflow
import luigi
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join

EXP_NAME = "Simplon"
ARTIFACT_PATH = "./mlruns/simplon"

if mlflow.get_experiment_by_name(EXP_NAME) is None:
    mlflow.create_experiment(name=EXP_NAME, artifact_location=ARTIFACT_PATH)
mlflow.set_experiment(EXP_NAME)


class RetrainModel(luigi.Task):
    date = luigi.DateParameter()

    def output(self):
        return luigi.LocalTarget(f"models/model_{self.date}.txt")

    def run(self):
        with mlflow.start_run():
            # Chargement des datasets
            df_old = pd.read_csv(join("data", "df_new.csv"))

            # Charger le préprocesseur
            #preprocessor_loaded = joblib.load(join("models", "preprocessor.pkl"))

            # preprocesser les data
            X, y, _ = preprocessing(df_old)

            # split data in train and test dataset
            X_train, X_test, y_train, y_test = split(X, y)

            # # create a new model
            model = create_nn_model(X_train.shape[1])

            # # entraîner le modèle
            model, _ = train_model(
                model, X_train, y_train, X_val=X_test, y_val=y_test
            )

            y_pred = model_predict(model, X_train)

            perf = evaluate_performance(y_train, y_pred)

            mlflow.log_metric("mse", perf["MSE"])
            mlflow.log_metric("mae", perf["MAE"])
            mlflow.log_metric("r2", perf["R²"])

            mlflow.sklearn.log_model(model, name="v2025_11", input_example=X_train)

            # # sauvegarder le modèle
            joblib.dump(model, join("models", "model_2025_11.pkl"))

        with self.output().open("w") as f:
            f.write(f"Model trained on {self.date} with mse={perf['mse']}\n")

    # draw_loss(hist)

    # # %% WARNING ZONE on test d'entrainer le modèle plus longtemps mais sur les mêmes données
    # model2, hist2 = train_model(
    #     model_2024_08, X_train, y_train, X_val=X_test, y_val=y_test
    # )
    # y_pred = model_predict(model_2024_08, X_test)
    # perf = evaluate_performance(y_test, y_pred)
    # print_data(perf, exp_name="exp 2")
    # draw_loss(hist2)
