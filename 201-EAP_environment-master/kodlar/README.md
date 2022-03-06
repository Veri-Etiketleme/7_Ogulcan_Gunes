# EAP_environment
Flask app to host our Employee Attrition Prediction Model on a server. Use python virtual environment to keep the setup clean.

mkdir project_name     #flask-app
cd project_name      #flask-app


install pip: sudo apt-get install python-pip

install virtual environment: sudo pip install virtualenv 

create virtual environment: virtualenv environment_name 

activate virtual environment : source environment_name/bin/activate

to deactivate environment: deactivate

Once virtualenv is activated

install flask: pip install Flask

install other libraries:
pip install numpy
pip install scipy
pip install scikit-learn
pip install treeinterpreter

run: python app.py
