from mlflow import *
from mlflow import client, models, artifacts, pyfunc, pytorch, tensorflow, onnx # Needs to be imported explicitly
import mlflow
import sys
import json
import os
import requests


__all__ = [
    "custom_set_details_json_path",
    "custom_set_experiment",
    "custom_get_current_run_id",
    "custom_start_run",
    "custom_load_run",
    "custom_send_email_notification",
    "custom_get_local_params"
]



def __show_error_and_exit__(message):
    print(f"\n\nERROR: {message}\n\n")
    sys.exit(1)


def __perform_checks__():
    experiment_name = __custom_mlflow_details__["experiment_details"]["experiment_name"].strip()
    run_name = __custom_mlflow_details__["experiment_details"]["run_name"].strip()
    run_description = __custom_mlflow_details__["experiment_details"]["run_description"]

    if experiment_name == "":
        __show_error_and_exit__("experiment_name cannot be empty!!!")
    
    if run_name == "":
        __show_error_and_exit__("run_name cannot be empty!!!")
    
    if run_description == "":
        __show_error_and_exit__("run_description cannot be empty!!!")

    # Check if experiment exists
    experiment = custom_set_experiment(experiment_name=experiment_name)
    if experiment == None:
        __show_error_and_exit__(f"Experiment \"{experiment_name}\" not found!!! Please create an experiment from the UI before proceeding.")


def custom_set_details_json_path(json_path="mlflow_details.json"):
    global __custom_mlflow_details__
    with open(json_path) as fobj:
        __custom_mlflow_details__ = json.load(fobj)

    os.environ["MLFLOW_TRACKING_URI"] = __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_URI"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_PASSWORD"]
    os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "true"
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR "] = "2" # Specifies the backoff increase factor between MLflow HTTP request failures (default: 2)
    os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_JITTER "] = "1.0" # Specifies the backoff jitter between MLflow HTTP request failures (default: 1.0)
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "7" # Specifies the maximum number of retries with exponential backoff for MLflow HTTP requests (default: 7)
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600" # Specifies the timeout in seconds for MLflow HTTP requests (default: 120)

    __perform_checks__()


def custom_set_experiment(experiment_name=None):
    global __custom_mlflow_details__
    experiment = None
    experiments = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
    # Don't use mlflow.get_experiment_by_name because it doesn't work for non-admin users
    if len(experiments) != 0:
        experiment = experiments[0]
    
    if experiment != None or __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_URI"] == "":
        experiment = mlflow.set_experiment(experiment_name=experiment_name)
    return experiment


def custom_get_current_run_id():
    global __custom_mlflow_details__
    experiment_name = __custom_mlflow_details__["experiment_details"]["experiment_name"].strip()
    run_name = __custom_mlflow_details__["experiment_details"]["run_name"].strip()
    experiment = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")[0]
    # Don't use mlflow.get_experiment_by_name because it doesn't work for non-admin users
    matched_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"run_name = '{run_name}'", output_format="list")
    run_id = None
    if len(matched_runs) > 0:
        run_id = matched_runs[0].info.run_id
    return run_id


def custom_start_run():
    global __custom_mlflow_details__
    run_name = __custom_mlflow_details__["experiment_details"]["run_name"].strip()
    description = __custom_mlflow_details__["experiment_details"]["run_description"]
    
    run_id = custom_get_current_run_id()
    
    if run_id != None: # Wants to create new run, but run with same name already exists
        __show_error_and_exit__(f"Cannot create new run!!! Run \"{run_name}\" already exists.")
    
    return mlflow.start_run(run_name=run_name, description=description, log_system_metrics=True)


def custom_load_run():
    global __custom_mlflow_details__
    run_name = __custom_mlflow_details__["experiment_details"]["run_name"].strip()
    description = __custom_mlflow_details__["experiment_details"]["run_description"]

    run_id = custom_get_current_run_id()
    
    if run_id == None: # Wants to continue run, but run doesn't exist
        __show_error_and_exit__(f"Cannot load run!!! Run \"{run_name}\" not found.")
    
    return mlflow.start_run(run_id=run_id, log_system_metrics=True)


def custom_send_email_notification(subject, body, additional_emails=""): # additional emails ending with dl-rl.com are supported, others are ignored as a safety measure.
    global __custom_mlflow_details__
    try:
        request_params = {"subject": subject, "body": body, "additional_emails": additional_emails}
        data = {}
        data["request_params"] = json.dumps(request_params)
        uri = __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_URI"]
        if not uri.endswith("/"):
            uri += "/"
        uri += "management/trigger_email_notification"
        response = requests.post(uri, data=data, auth=(__custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_USERNAME"], __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_PASSWORD"]))
        return response.text # Will return "done" if successful
    except Exception as e:
        return str(e)


def custom_get_local_params():
    global __custom_mlflow_details__
    return __custom_mlflow_details__["params"]


