from flask.templating import render_template
import joblib
import pandas as pd
from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    CRIM = float(request.form['CRIM'])
    ZN = float(request.form['ZN'])
    INDUS = float(request.form['INDUS'])
    CHAS = float(request.form['CHAS'])
    NOX = float(request.form['NOX'])
    RM = float(request.form['RM'])
    AGE = float(request.form['AGE'])
    DIS = float(request.form['DIS'])
    RAD = float(request.form['RAD'])
    TAX = float(request.form['TAX'])
    PTRATIO = float(request.form['PTRATIO'])
    B = float(request.form['B'])
    LSTAT = float(request.form['LSTAT'])

    data = pd.DataFrame([[CRIM, ZN,	INDUS, CHAS, NOX, RM,	AGE, DIS, RAD, TAX,	PTRATIO, B, LSTAT]],
                        columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])

    lin_reg = joblib.load('model/Linear-Regression-Model.pkl')
    numerical_transformer = joblib.load('model/numerical_transformer.pkl')
    column_transformer = joblib.load('model/column_transformer.pkl')

    prepared_data = column_transformer.transform(data)
    output = lin_reg.predict(prepared_data)
    final_output = numerical_transformer.inverse_transform(output)
    final_output = '{:.2f}'.format(final_output[0, 0])


    return render_template('index.html', result=final_output)


if __name__ == '__main__':
    app.run(debug=True)
