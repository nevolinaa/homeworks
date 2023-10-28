"""
В рамках подготовки домашнего задания все имена переменных могут быть любыми (когда тестируете у себя). 
Но при сдаче домашнего задания на проверку просьба не менять следующие пункты:
  - BUCKET оставить как есть в этом шаблоне;
  - EXPERIMENT_NAME и DAG_ID оставить как есть (ссылками на переменную NAME);
  - имена коннекторов: pg_connection и s3_connection;
  - данные должны читаться из таблицы с названием california_housing;
  - данные на S3 должны лежать в папках {NAME}/datasets/ и {NAME}/results/.
"""
import json
import logging
import mlflow
import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
import os

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "nevolinaa" # TO-DO: Вписать свой ник в телеграме
BUCKET = "lizvladi-mlops"
FEATURES = [
    "LnMedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME

os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net'
os.environ['MLFLOW_TRACKING_URI'] = 'postgresql://mlflow:mlflow@localhost:5432/mlflow_db'

models = dict(zip(["rf", "lr", "hgb"], 
                  [RandomForestRegressor(), 
                   LinearRegression(), 
                   HistGradientBoostingRegressor()]
                 )) # TO-DO: Создать словарь моделей

default_args = {
    "owner" : "Arina_Nevolina",
    "email_on_failure" : False,
    "email_in_retry" : False,
    "retries" : 3,
    "retry_delay" : timedelta(minutes=1)
    # TO-DO: Заполнить своими данными: настроить владельца и политику retries.
}

dag = DAG(dag_id=DAG_ID,
          default_args=default_args,
          schedule_interval = "0 1 * * *",
          start_date = days_ago(2),
          catchup = False,
          tags = ["mlops"]
          # TO-DO: Заполнить остальными параметрами.
          )


def init() -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать start_tiemstamp, run_id, experiment_name, experiment_id.
    metrics = {}
    metrics["start_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")
    metrics["experiment_name"] = NAME

    # TO-DO 2 mlflow: Создать эксперимент с experiment_name=NAME. 
    # Добавить проверку на уже созданный эксперимент!

    results = mlflow.search_experiments(filter_string=f"name = '{EXPERIMENT_NAME}'")
    if results == []:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=f's3://{BUCKET}/{EXPERIMENT_NAME}')
        mlflow.set_experiment(EXPERIMENT_NAME)
    else:
        current_experiment=dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
        experiment_id=current_experiment['experiment_id']

    metrics["experiment_id"] = experiment_id

    # TO-DO 3 mlflow: Создать parent run.
    with mlflow.start_run(run_name="PARENT_RUN", 
                          experiment_id = experiment_id, 
                          description = "parent") as parent_run:
        metrics["run_id"] = mlflow.active_run().info.run_id

    return metrics


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_download_start.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="init")
    metrics["data_download_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    pg_hook = PostgresHook("pg_connection")
    con = pg_hook.get_conn()

    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    # TO-DO 3 Postgres: Прочитать данные.
    data = pd.read_sql_query("SELECT * FROM california_housing", con)

    # TO-DO 4 Postgres: Сохранить данные на S3 в формате pickle в папку {NAME}/datasets/.
    file_name = f"{NAME}/datasets/california_housing.pkl"
    
    pickle_byte_obj = pickle.dumps(data)
    resource.Object(BUCKET, file_name).put(Body=pickle_byte_obj)

    # TO-DO 5 metrics: В этом шаге собрать data_download_end.
    metrics["data_download_end"] = datetime.now().strftime("%Y%m%d %H:%M")
    _LOG.info("Data download finished.")
    
    return metrics


# Создаем функцию для удаления выбросов
def remove_outliers(df, cols):
    for col in cols:
        col_99_percentile = np.percentile(df[col], 99)
        df[col] = np.where(df[col] > col_99_percentile, np.nan, df[col])
    return df

def prepare_data(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: Собрать data_preparation_start.
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="get_data")
    metrics["data_preparation_start"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 2 connections: Создать коннекторы.
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    # TO-DO 3 S3: Прочитать данные с S3.
    file_name = f"{NAME}/datasets/california_housing.pkl"
    file = s3_hook.download_file(key=file_name, bucket_name=BUCKET)
    data = pd.read_pickle(file)

    # TO-DO 4: Сделать препроцессинг.
    # работа с выбросами
    cols = ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    remove_outliers(data, cols)
    data[cols] = data[cols].fillna(data[cols].median())

    # приводим распределение переменной MedInc к нормальному
    data["LnMedInc"] = np.log(data["MedInc"])
    data.drop(columns="MedInc", axis=1, inplace=True)

    X, y = data[FEATURES], data[TARGET]

    # TO-DO 5: Разделить данные на train/test.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)

    # TO-DO 6: Подготовить 4 обработанных датасета.
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    # Сохранить данные на S3 в папку {NAME}/datasets/.
    for name, data in zip(["X_train", "X_test", "y_train", "y_test"],
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(BUCKET,
                        f"{NAME}/datasets/{name}.pkl").put(Body=pickle_byte_obj)
    
    # TO-DO 7 metrics: собрать data_preparation_end.
    metrics["data_preparation_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    _LOG.info("Data preparation finished.")

    return metrics


def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    # TO-DO 1: Обучить модель.
    model.fit(X_train, y_train)
    
    # TO-DO 2: Сделать predict.
    predictions = model.predict(X_test)
    
    # TO-DO 3: Сохранить результаты обучения с помощью MLFlow.
    # Посчитать метрики
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared = False)
    mae = mean_absolute_error(y_test, predictions)

    # сохранить метрики
    mlflow.log_metric('r2_score', r2)
    mlflow.log_metric('rmse', rmse)
    mlflow.log_metric('mae', mae)

    # сохранить модель
    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(model, name, signature=signature)

def train_model(**kwargs) -> Dict[str, Any]:
    ti = kwargs["ti"]
    metrics = ti.xcom_pull(task_ids="prepare_data")
    m_name = kwargs["model_name"]

    # TO-DO 1 connections: Создать коннекторы.
    s3_hook = S3Hook("s3_connection")

    # TO-DO 2 S3: Прочитать данные с S3 из папки {NAME}/datasets/.
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"{NAME}/datasets/{name}.pkl",
                                     bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # TO-DO 3: Обучить модели и залогировать обучение с помощью MLFlow.
    # И собрать для metrics f"{m_name}_train_start" и f"{m_name}_train_end"
    model = models[m_name]
    metrics[f"{m_name}_train_start"] = datetime.now().strftime("%Y%m%d %H:%M")
    model.fit(data["X_train"], data["y_train"])

    prediction = model.predict(data["X_test"])
    metrics[f"{m_name}_train_end"] = datetime.now().strftime("%Y%m%d %H:%M")

    metrics[f"{m_name}_r2_score"] = r2_score(data["y_test"], prediction)
    metrics[f"{m_name}_rmse"] = mean_squared_error(data["y_test"],
                                                   prediction)**0.5
    metrics[f"{m_name}_mae"] = mean_absolute_error(data["y_test"],
                                                     prediction)
    
    with mlflow.start_run(run_id=metrics["run_id"], experiment_id=metrics["experiment_id"]) as PARENT_RUN:
        with mlflow.start_run(run_name=m_name, experiment_id=metrics["experiment_id"], nested=True) as child_run:
             train_mlflow_model(model, m_name, data["X_train"], data["X_test"], data["y_train"], data["y_test"])

    return metrics


def save_results(**kwargs) -> None:
    ti = kwargs["ti"]
    date = datetime.now().strftime("%Y_%m_%d_%H")

    # Создать коннекторы.
    s3_hook = S3Hook("s3_connection")
    session = s3_hook.get_session("ru-central1")
    resource = session.resource("s3")

    # Собрать информацию в один словарь
    metrics_itog = {} 
    
    for model_name in models.keys():
        metrics = ti.xcom_pull(task_ids=f"train_{model_name}")
        metrics_itog.update(metrics)

    # TO-DO 1 metrics: В этом шаге собрать end_timestamp.
    metrics_itog["end_timestamp"] = datetime.now().strftime("%Y%m%d %H:%M")

    # TO-DO 1: сохранить результаты обучения на S3 в файл {NAME}/results/{date}.json.
    file_name = f"{NAME}/results/{date}.json"
    json_byte_object = json.dumps(metrics_itog)
        
    resource.Object(BUCKET, file_name).put(Body=json_byte_object)



#################################### INIT DAG ####################################

task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)

task_get_data = PythonOperator(task_id="get_data",
                               python_callable=get_data_from_postgres,
                               dag=dag,
                               provide_context=True)

task_prepare_data = PythonOperator(task_id="prepare_data",
                                   python_callable=prepare_data,
                                   dag=dag,
                                   provide_context=True)

task_train_models = [PythonOperator(task_id=f"train_{model_name}",
                                    python_callable=train_model,
                                    dag=dag,
                                    provide_context=True,
                                    op_kwargs={"model_name": model_name}) for model_name in models.keys()]

task_save_results = PythonOperator(task_id="save_results",
                                   python_callable=save_results,
                                   dag=dag,
                                   provide_context=True)

# TO-DO: Прописать архитектуру DAG'a.
task_init >> task_get_data >> task_prepare_data >> task_train_models >> task_save_results