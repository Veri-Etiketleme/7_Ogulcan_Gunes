import urllib3, requests, json

# retrieve your wml_service_credentials_username, wml_service_credentials_password, and wml_service_credentials_url from the
# Service credentials associated with your IBM Cloud Watson Machine Learning Service instance

wml_credentials={
"password": "c3e45a27-fd91-40a0-a61b-fde58d7d4a71",
"url": "https://eu-gb.ml.cloud.ibm.com",
"username": "39b756b5-9e5a-4372-ac56-90484937f0ca"
}

headers = urllib3.util.make_headers(basic_auth='{username}:{password}'.format(username=wml_credentials['username'], password=wml_credentials['password']))
url = '{}/v3/identity/token'.format(wml_credentials['url'])
response = requests.get(url, headers=headers)
mltoken = json.loads(response.text).get('token')

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
array_of_values_to_be_scored = [35, 'Travel_Rarely', 1102, 'Sales', 1, 2, 'Bachelor', 94,'Female',5, 7, 2, 'Sales', 5, 'Single', 3000, 30, 1, 'Yes', 70, 25, 3, 1, 1, 17, 15, 1, 1, 1, 1 ]
another_array_of_values_to_be_scored = [41, 'Travel_Frequently', 1200, 'Research & Development', 1, 2, 'Bachelor', 94,'Male',5, 7, 2, 'Research Scientist', 5, 'Married', 3000, 30, 1, 'No', 70, 25, 3, 1, 1, 17, 15, 1, 1, 1, 1 ]
payload_scoring = {"fields": ["Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"], "values": [array_of_values_to_be_scored, another_array_of_values_to_be_scored]}

response_scoring = requests.post('https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/3c7b8e7e-8cb7-4bf6-8cf3-da5722f20baf/deployments/0f9b1a75-7424-409d-a65a-0f3c983f029f/online', json=payload_scoring, headers=header)
print("Scoring response")
#print(json.loads(response_scoring.text))

parsed = json.loads(response_scoring.text)

for value in parsed['values']:
    fields = value[:30]
    confidence = value[32]
    prediction = value[34]
    values = value[35]
    print("\n Employee Attrition: {}\n\tConfidence: {}\n\tFields {}".format(prediction,zip(confidence, values), fields) )