# USAGE
# Start the server:
# 	python run_server.py
# Submit a request via Python:
# python step3_mlinbusiness_cp.py

# import the necessary packages
import dill
import pandas as pd
import os
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

dill._dill._reverse_typemap['ClassType'] = type

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)['model']
    print(model)


# для работы в докер контейнере
modelpath = "/app/app/models/gb_pipeline.dill"

# для работы на локальном компьютере
# modelpath = "models/gb_pipeline.dill"

load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to prediction of loan return process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        request_json = flask.request.get_json()
        # check data is returned
        if len(request_json):
            logger.info(f"{dt} Data: {request_json}")
            try:
                preds = model.predict_proba(pd.DataFrame(request_json, index=[0]))
            except AttributeError as e:
                logger.warning(f'{dt} Exception: {str(e)}')
                data['predictions'] = str(e)
                data['success'] = False
                return flask.jsonify(data)

            data["predictions"] = preds[:, 1][0]
            # indicate that the request was a success
            data["success"] = True
        else:
            logger.info(f"{dt} Data: json data is empty")

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
