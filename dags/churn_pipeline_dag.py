from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import pipeline steps
from src.data_ingestion.ingest import run as run_ingest
from src.features.build_features import run as run_features
from src.models.train_churn import run as run_train
from src.models.train_uplift import run as run_train_uplift
from src.models.segment import run as run_segment
from src.models.retention_simulator import run as run_simulate

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction, segmentation, and simulation DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['churn', 'mlops'],
) as dag:

    t1 = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_ingest,
    )

    t2 = PythonOperator(
        task_id='feature_engineering',
        python_callable=run_features,
    )

    t3 = PythonOperator(
        task_id='model_training',
        python_callable=run_train,
    )

    t3_uplift = PythonOperator(
        task_id='uplift_model_training',
        python_callable=run_train_uplift,
    )

    t4 = PythonOperator(
        task_id='customer_segmentation',
        python_callable=run_segment,
    )

    t5 = PythonOperator(
        task_id='retention_simulation',
        python_callable=run_simulate,
    )

    # Define dependencies
    t1 >> t2 >> t3 >> t3_uplift >> t4 >> t5
