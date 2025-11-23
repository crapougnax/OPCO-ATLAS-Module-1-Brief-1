import mlflow
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict
import pandas as pd
import joblib
from os.path import join as join

with mlflow.start_run():
    mlflow.set_tracking_uri("http://localhost:5555")

    # Chargement des datasets
    df_old = pd.read_csv(join("data", "df_old.csv"))
    df_new = pd.read_csv(join("data", "df_new.csv"))

    # Definition des paramètres
    test_size = 0.10
    random_state = 84
    dataset_label = "combined"

    model_name = (
        "v2025_11_"
        + dataset_label
        + "_ts"
        + str(int(test_size * 100))
        + "_rs"
        + str(random_state)
    )

    ds = pd.concat([df_old, df_new])

    # Create a Dataset object
    dataset = mlflow.data.from_pandas(ds)
    mlflow.log_input(dataset, context=dataset_label)

    # preprocesser les data
    X, y, _ = preprocessing(ds)

    # split data in train and test dataset
    X_train, X_test, y_train, y_test = split(
        X, y, test_size=test_size, random_state=random_state
    )

    # # create a new model
    model = create_nn_model(X_train.shape[1])

    # # entraîner le modèle
    model, hist2 = train_model(model, X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred = model_predict(model, X_train)

    perf = evaluate_performance(y_train, y_pred)

    mlflow.log_metric("mse", perf["MSE"])
    mlflow.log_metric("mae", perf["MAE"])
    mlflow.log_metric("r2", perf["R²"])

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    mlflow.sklearn.log_model(
        model,
        name=model_name,
        input_example=X_train,
    )

    # # sauvegarder le modèle
    joblib.dump(model, join("models", model_name))

    draw_loss(hist2)
