# MASproject

A Multiagent System (MAS) which follows a group of keywords and classify the users who are tweeting them in two distinct classes: Human or Bot.

  

## Installation

  

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the MASproject. You can install the dependencies running the install script:

  

```
bash install.sh 
```  

Or running the comands separately
```
source bin/activate
pip3 install -r requirements.txt 
```

## Usage

Twitter keys and tokens are needed to use the Twitter API, and consequently for using the MASproject. To obtain them access https://developer.twitter.com/en/apps register your app, and get your api keys (and tokens). After this create your 'keys.txt' file running the following command, with your credentials:

  
  

```
echo  "API key@API secret key@Access token@access token secret" >  "data/keys.txt"
```

With 'keys.txt' created you will be able to run the MASproject using the 'run.sh' script:

```
bash run.sh
```
  

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.