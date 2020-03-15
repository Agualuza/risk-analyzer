from sklearn.naive_bayes import MultinomialNB

import csv
import numpy as np
import pickle

import random

def dataGenerator():
  rows = [["persona_id","category_id","payment","bill","product_price","evaluation"]]

  for i in range(200000):
    pid = random.randint(1,5)
    cid = random.randint(1,11)
    pp = random.randint(50,15000)
    rows.append(analyze(pid,cid,pp))

  with open('dataset3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)

def analyze(pid, cid, pp):
    evaluations = ["RB", "RM", "RE", "YB", "YM", "YE", "GB", "GM", "GE"]
    payment = random.randint(980, 25000)

    if pid == 1:
        bill = 0.3 * payment
    elif pid == 2:
        bill = 0.5 * payment
    elif pid == 3:
        bill = 0.7 * payment
    elif pid == 4:
        bill = 1.0 * payment
    else:
        bill = 1.2 * payment

    factors = [1.2, 1.1, 1.0, 0.9, 0.8]
    categoriesTypes = ["L", "L", "L", "F", "F", "F", "N", "N", "N", "P", "P"]

    balance = (payment * 0.7) - bill
    balance = max(balance, 400)

    response = (balance / pp) * factors[pid - 1]

    evaluation = calculateEvaluation(response)
    evaluation = getBestEvalAllowed(evaluations[evaluation], evaluation, categoriesTypes[cid - 1], pid)

    resp = []
    resp.append(pid)
    resp.append(cid)
    resp.append(payment)
    resp.append(bill)
    resp.append(pp)
    resp.append(evaluations[evaluation])

    return resp

def calculateEvaluation(r):
  if (r >= 1.0) :
    return 8
  elif (r >= 0.9):
    return 7
  elif (r >= 0.8):
    return 6
  elif (r >= 0.7):
    return 5
  elif (r >= 0.6):
    return 4
  elif (r >= 0.5):
    return 3
  elif (r >= 0.4):
    return 2
  elif (r >= 0.3):
    return 1
  else:
    return 0

def getBestEvalAllowed(r,rindex,ct,pid):
  if (pid == 1):
    return rindex
  elif (pid == 2):
    if (ct == "L" and (r == "GE" or r == "GM")):
      return 6
  elif (pid == 3):
    if ((ct == "L" or ct == "F") and (r == "GE" or r == "GM" or r == "GB" or r == "YE")):
      return 4
  elif (pid == 4):
    if ((ct != "P") and (r == "GE" or r == "GM" or r == "GB" or r == "YE" or r == "YM")):
      return 3
  elif (pid == 5):
    if ((ct != "P") and (r == "GE" or r == "GM" or r == "GB" or r == "YE" or r == "YM" or r == "YB")):
      return 2
  
  return rindex

def loadDataSet():
    data = []
    response = []

    file = open('dataset3.csv', 'r')
    reader = csv.reader(file)
    for persona_id, category_id,payment, bill, product_price, evaluation in reader:
        data.append([persona_id, category_id,payment, bill, product_price])
        response.append(evaluation)

    return data, response

def train(model,dataset,response):
  data = np.array(dataset).astype(np.float)
  model.fit(data, response)
  pickle.dump(model, open('model.sav', 'wb'))

def run(pid,cid,payment,bill,pp):
    # Methods to train model
    # dataset , response = loadDataSet()
    # model = MultinomialNB()
    # dataset.pop(0)
    # response.pop(0)
    # train(model,dataset,response)

    model = pickle.load(open('model.sav', 'rb'))

    misterioso = [[pid,cid,payment,bill,pp]]
    r = model.predict(misterioso)

    json_response = {}
    json_response["status"] = "NOK"
    json_response["response"] = None
    if r[0]:
        json_response["status"] = "OK"
        json_response["response"] = r[0]
    return json_response

