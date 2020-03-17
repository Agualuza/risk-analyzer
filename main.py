from flask import Flask,jsonify,request
from flask_cors import CORS
from methods import run,runLogisticRegression,runAdaBoost
import os

app = Flask(__name__)
cors = CORS(app)
port = os.getenv('PORT') if os.getenv('PORT') else "8000"

@app.route("/")
def main():
    pp = float(request.args.get('product_price'))
    pid = int(request.args.get('persona_id'))
    cid = int(request.args.get('category_id'))
    payment = float(request.args.get('payment'))
    bill = float(request.args.get('bill'))

    response = run(pid,cid,payment,bill,pp)
    return jsonify(response)


@app.route("/boost")
def main_boost():
    pp = float(request.args.get('product_price'))
    pid = int(request.args.get('persona_id'))
    cid = int(request.args.get('category_id'))
    payment = float(request.args.get('payment'))
    bill = float(request.args.get('bill'))

    response = runAdaBoost(pid, cid, payment, bill, pp)
    return jsonify(response)


@app.route("/lregression")
def main_regression():
    pp = float(request.args.get('product_price'))
    pid = int(request.args.get('persona_id'))
    cid = int(request.args.get('category_id'))
    payment = float(request.args.get('payment'))
    bill = float(request.args.get('bill'))

    response = runLogisticRegression(pid, cid, payment, bill, pp)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port)