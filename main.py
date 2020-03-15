from flask import Flask,jsonify,request
from methods import run
import os

app = Flask(__name__)
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

if __name__ == "__main__":
    app.run(port=port)