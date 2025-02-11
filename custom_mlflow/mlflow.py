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

    __perform_checks__()


def custom_set_experiment(experiment_name=None, experiment_id=None):
    global __custom_mlflow_details__
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment = mlflow.get_experiment(experiment_id)
        if experiment.name == "Default":
            experiment = None
    
    if experiment != None or __custom_mlflow_details__["credentials"]["MLFLOW_TRACKING_URI"] == "":
        experiment = mlflow.set_experiment(experiment_name=experiment_name, experiment_id=experiment_id)
    return experiment


def custom_get_current_run_id():
    global __custom_mlflow_details__
    experiment_name = __custom_mlflow_details__["experiment_details"]["experiment_name"].strip()
    run_name = __custom_mlflow_details__["experiment_details"]["run_name"].strip()
    experiment = mlflow.get_experiment_by_name(experiment_name)
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


def custom_send_email_notification(subject, body):
    global __custom_mlflow_details__
    try:
        request_params = {"subject": subject, "body": body}
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


