from flask import Flask, request, render_template, url_for
import numpy as np
import pickle

model = pickle.load(open("decisiontreeclassifier.pkl", "rb"))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data1 = request.form["sepallength"]
    data2 = request.form["sepalwidth"]
    data3 = request.form["petallength"]
    data4 = request.form["petalwidth"]

    pred_values = np.array([[data1, data2, data3, data4]])
    pred = model.predict(pred_values)

    return render_template("index.html", prediction=pred)


# if __name__ == "__main__":
#     app.run(debug=True)
