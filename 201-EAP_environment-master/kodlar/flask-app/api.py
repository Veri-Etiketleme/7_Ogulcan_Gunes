import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc
import pandas as pd

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
        sample=[]
        #print request.form
        satisfaction=request.form.get('Satisfaction')
        sample.append(float(satisfaction))
        evaluation=request.form.get('Evaluation')
        sample.append(float(evaluation))
        projectCount=request.form.get('Project Count')
        sample.append(int(projectCount))
        averageMonthlyHours=request.form.get('Average Monthly Hours')
        sample.append(int(averageMonthlyHours))
        yearsAtCompany=request.form.get('Years At Company')
        sample.append(int(yearsAtCompany))
        workAccident=request.form.get('Work Accident')
        sample.append(int(workAccident))
        promotion=request.form.get('Promotion')
        sample.append(int(promotion))
        salary=request.form.get('Salary')
        sample.append(int(salary))
        department=request.form.get('Department')
        sample.append(department)
        sampleDf= pd.DataFrame(sample)
        if len(sample)<9: return render_template('index.html', label1="Missing data or incorrect data entered")
        # make prediction
        prediction = model.predict(sampleDf.T)
        confidence = model.predict_proba(sampleDf.T)
        if prediction ==1:
            label2 = "Confidence Level : "+str(round(confidence[0][1],2))
        else:
            label2 = "Confidence Level : "+str(round(confidence[0][0],2))
        if prediction[0]==1:
            return render_template('index.html', label1="This Employee is at a high risk of leaving the Company", label2=label2)
        else:
            return render_template('index.html', label1="This Employee is not at risk of leaving the Company", label2=label2)


if __name__ == '__main__':
    # load ml model
    model = joblib.load('model.pkl')
    # start api
    app.run(host='0.0.0.0', port=8000, debug=True)
