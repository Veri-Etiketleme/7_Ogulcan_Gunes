# set base image (host OS)
FROM python:3.7

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# copy the content of the local src directory to the working directory
COPY data/ ./data

# copy the content of the local src directory to the working directory
COPY classifiers/ ./classifiers

EXPOSE 5000/tcp
EXPOSE 8000

# command to run on container start
CMD ["pade","start-runtime","--port", "20000", "subscribe.py"] 