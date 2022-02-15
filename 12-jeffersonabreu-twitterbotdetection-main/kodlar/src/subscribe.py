from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaSubscribeProtocol, TimedBehaviour
from sys import argv
from pade.acl.messages import ACLMessage, AID

import tweepy
from queue import Queue
from threading import Thread
import os
import time
import datetime
import random
import ast
import pickle
import pandas as pd

import json

# import classifier


class PublisherProtocolColector(FipaSubscribeProtocol):
    def __init__(self, agent):
        super(PublisherProtocolColector, self).__init__(
            agent, message=None, is_initiator=False)

    def handle_subscribe(self, message):
        self.register(message.sender)
        display_message(self.agent.aid.name, '{} from {}'.format(
            message.content, message.sender.name))
        resposta = message.create_reply()
        resposta.set_performative(ACLMessage.AGREE)
        resposta.set_content('Subscribe message accepted')
        self.agent.send(resposta)
        self.agent.registered = self.agent.registered + 1

        if self.agent.registered == 3:
            self.agent.timed.ready = True

    def handle_cancel(self, message):
        self.deregister(message.sender)
        display_message(self.agent.aid.name, message.content)

    def notify(self, message):
        super(PublisherProtocolColector, self).notify(message)


class Time(TimedBehaviour):
    def __init__(self, agent, notify):
        super(Time, self).__init__(agent, 0.01)
        self.notify = notify
        self.inc = 0
        self.ready = False

    def on_time(self):
        super(Time, self).on_time()
        message_body = str(self.getUserData())
        if message_body != '':
            message = ACLMessage(ACLMessage.INFORM)
            message.set_protocol(ACLMessage.FIPA_SUBSCRIBE_PROTOCOL)
            message.set_content(message_body)
            self.notify(message)
            self.inc += 0.1

    def getUserData(self):
        file_size = os.stat('data/buffer.json').st_size
        lines = []

        if self.ready:
            if file_size == 0:
                display_message(self.agent.aid.name, 'The file is empty')
                time.sleep(10)
                return ''
            else:
                f = open('data/buffer.json', 'r')
                lines = f.readlines()
                f.close()
                f = open('data/buffer.json', 'w')
                f.write(''.join(lines[1:]))
                f.close()
                
                tweet = lines[0]
                
                f = open('data/tweets_data/tweets_data.json', 'w+')
                f.write(str(tweet))
                f.close()

                return tweet
        else:
            return ''


class TimeTT(TimedBehaviour):
    def __init__(self, agent):
        super(TimeTT, self).__init__(agent, 1800)
        self.agent = agent

    def on_time(self):
        super(TimeTT, self).on_time()
        self.agent.updateFilter()


class MyStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        self.saveUserData(str(status._json))

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def __init__(self, q=Queue()):
        super().__init__()
        self.q = q
        for i in range(4):
            t = Thread(target=self.do_stuff)
            t.daemon = True
            t.start()

    def do_stuff(self):
        while True:
            self.q.get()
            self.q.task_done()

    def saveUserData(self, data):
        FILE = open("data/buffer.json", "a+")
        FILE.write(data + '\n')
        FILE.close()

    def backupUserData(self, data):
        FILE = open("data/buffer.json", "r")
        data = FILE.read()
        FILE.close()

        ts = str(time.time()).replace(".", "") + '.json'

        FILE = open("data/bkp/" + ts, "a")
        FILE.write(data)
        FILE.close()

        self.cleanUserData()

    def cleanUserData(self):
        FILE = open("data/buffer.json", "w")
        FILE.close()


class AgentReferee(Agent):

    def __init__(self, aid, participants, number_of_classifiers):
        self.participants = participants
        self.classifiers = list(
            filter(lambda x: 'agent_classifier' in x, self.participants))

        self.RF_name = list(filter(lambda x: '_RF' in x, self.classifiers))[0]
        self.SVM_name = list(
            filter(lambda x: '_SVM' in x, self.classifiers))[0]
        self.NB_name = list(filter(lambda x: '_NB' in x, self.classifiers))[0]
        self.number_of_classifiers = number_of_classifiers
        super(AgentReferee, self).__init__(aid)
        self.list_of_classification = list()

        self.startDecisionsFile()

    def startDecisionsFile(self):
        FILE = open("data/decisions/decisions.csv", "w")
        FILE.close()

        FILE = open("data/decisions/decisions.csv", "a")
        FILE.write(
            '"id","RF_vote","RF_vote_proba","NB_vote","NB_vote_proba","SVM_vote","SVM_vote_proba","SMA_decision","SMA_decision_proba", "tweet_creation_ts", "sma_decision_ts"\n')
        FILE.close()

    def react(self, message):
        if str(message.sender.name) in participants:
            display_message(self.aid.name, str(
                message.sender.name) + str(message.content))
            content = str(message.content).split('¢')
            self.decide(str(message.sender.name),
                        content[0], content[1], content[2], content[3])

    def decide(self, agent_id, user_id, user_class, user_class_proba, created_ts):
        has_user_id = list(filter(lambda x: x['user_id'] == str(
            user_id),  self.list_of_classification))

        if len(has_user_id):
            index = self.list_of_classification.index(list(
                filter(lambda x: x['user_id'] == str(user_id), self.list_of_classification))[0])
            self.list_of_classification[index][str(agent_id)] = user_class
            self.list_of_classification[index][str(
                agent_id)+'_proba'] = user_class_proba
        else:
            element = {
                'user_id': str(user_id),
                'created_ts': str(created_ts)
            }
            element[str(agent_id)] = user_class
            element[str(agent_id)+'_proba'] = user_class_proba
            self.list_of_classification.append(element)
        print('\n\n\nchecking')
        self.checkForAvailableClassifications()

    def checkForAvailableClassifications(self):
        
        for x in self.list_of_classification:
            decision = 0
            probability = 0
            if (len(x) == ((self.number_of_classifiers * 2) + 2)):

                index = self.list_of_classification.index(x)
                for agent_classifier in self.classifiers:
                    decision += int(x[str(agent_classifier)])/2
                isBot = True if decision > (
                    number_of_classifiers/2) else False

                for agent_classifier in self.classifiers:
                    probability += float(x[str(agent_classifier)+'_proba'])
                sma_proba = probability/self.number_of_classifiers

                now = datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc)

                self.list_of_classification.pop(index)
                
                param = {
                    'id': str(x['user_id']), 
                    'RF': x[str(self.RF_name)], 
                    'RF_proba': x[str(self.RF_name)+'_proba'], 
                    'NB': x[str(self.NB_name)], 
                    'NB_proba': x[str(self.NB_name)+'_proba'], 
                    'SVM': x[str(self.SVM_name)], 
                    'SVM_proba': x[str(self.SVM_name)+'_proba'], 
                    'isBot': isBot, 
                    'isBot_proba': sma_proba, 
                    'created_ts': str(x['created_ts']), 
                    'decision_ts': str(now),
                }

                display_message(
                    self.aid.name, x['user_id'] + ' is bot? ' + str(isBot) + ' probability ' + str(sma_proba))

                print('\n\nsave')
                self.saveDecision(param)
                
                if isBot:
                    # TODO report
                    pass

    def saveDecision(self, params):
        st_num_isBot = '1' if params['isBot'] else '0'
        FILE = open("data/decisions/decisions.csv", "a")

        FILE.write('"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(
            params['id'], params['RF'], params['RF_proba'], params['NB'], params['NB_proba'], params['SVM'], params['SVM_proba'], st_num_isBot, params['isBot_proba'], params['created_ts'], params['decision_ts']))
        
        
        FILE.close()


class AgentColector(Agent):
    # agent attributes
    keys = []
    api = None
    myStreamListener = None
    myStream = None
    inc = 0
    isOffline = False
    registered = 0
    brazil = 23424768
    trending = []

    def __init__(self, aid, number_of_classifiers, offline=False):
        super(AgentColector, self).__init__(aid)
        self.number_of_classifiers = number_of_classifiers
        self.myStreamListener = MyStreamListener()
        self.protocol = PublisherProtocolColector(self)
        self.behaviours.append(self.protocol)
        self.myStreamListener.cleanUserData()
        self.isOffline = offline
        self.run(self.isOffline)

    def run(self, offline):
        if (offline == True):
            self.populateUserData()
        else:
            self.startStream()

    def startStream(self):
        self.inc = 0
        self.inc += 0.1

        self.keys = self.getKeys()
        self.api = {}

        for key in self.keys:
            self.api = self.authenticate(key[0], key[1], key[2], key[3])

        self.myStream = tweepy.Stream(
            auth=self.api["auth"], listener=self.myStreamListener)

        self.startTimeBehavior()
        self.updateFilter()

    def updateFilter(self):
        self.trending = self.returnTrendings(self.api['api'], self.brazil)
        self.startFiltering(self.trending)

    def startFiltering(self, topics):
        FILE = open("data/logs/topics.txt", "a+")
        now = datetime.datetime.fromtimestamp(time.time(), datetime.timezone.utc)
        FILE.write("{" + 'trending: "{}", date: "{}"'.format(str(topics), str(now)) + "}\n")
        FILE.close()

        self.myStream.disconnect()
        self.myStream.filter(track=topics, is_async=True)

    def returnTrendings(self, api, place):
        trends = api.trends_place(place)
        trends_list = []

        for x in trends[0]['trends']:
            trends_list.append(x['name'])
        return trends_list
        self.startTimeBehavior()

    def startTimeBehavior(self):
        self.timed = Time(self, self.protocol.notify)
        self.behaviours.append(self.timed)
        self.timedtt = TimeTT(agent=self)
        self.behaviours.append(self.timedtt)

    def getKeys(self):
        FILE = open("data/keys.txt", "r")
        keys0 = FILE.read()
        keys1 = []
        FILE.close()

        keys0 = keys0.split("\n")

        for key in keys0:
            keys1.append(key.split("@"))
        return [keys1[0]]

    def authenticate(self, CONSUMER_TOKEN, CONSUMER_SECRET, ACCESS_KEY, ACCESS_SECRET):
        auth = tweepy.OAuthHandler(CONSUMER_TOKEN, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_KEY, ACCESS_SECRET)

        # Construct the API instance
        api = tweepy.API(
            auth,
            wait_on_rate_limit=True,
            wait_on_rate_limit_notify=True,
            compression=True,
            retry_count=9080,
            retry_delay=15,
        )

        dct = {}
        dct["api"] = api
        dct["auth"] = auth
        return dct

    def populateUserData(self):
        genuine_users = pd.read_csv(
            'data/datasets/test_set/genuine_X_test.csv')
        social1_users = pd.read_csv(
            'data/datasets/test_set/traditional_X_test.csv')

        genuine_users['json'] = genuine_users.apply(
            lambda x: x.to_json(), axis=1)
        social1_users['json'] = social1_users.apply(
            lambda x: x.to_json(), axis=1)

        for index, row in genuine_users.iterrows():
            obj = str('{"user": #}'.replace(
                '#', row['json'].replace("null", "None")))
            self.myStreamListener.saveUserData(obj)

        for index, row in social1_users.iterrows():
            obj = str('{"user": #}'.replace(
                '#', row['json'].replace("null", "None")))
            self.myStreamListener.saveUserData(obj)
        self.startTimeBehavior()


class SubscriberProtocolClassifier(FipaSubscribeProtocol):

    def __init__(self, agent, message):
        super(SubscriberProtocolClassifier, self).__init__(
            agent, message, is_initiator=True)

    def handle_agree(self, message):
        display_message(self.agent.aid.name, str(
            message.content))

        message = ACLMessage(ACLMessage.INFORM)
        message.set_protocol(ACLMessage.FIPA_REQUEST_PROTOCOL)
        message.add_receiver(agent_colector.aid)
        message.set_content('ready')
        self.agent.send(message)

    def handle_inform(self, message):
        x = ast.literal_eval(str(message.content))
        classy = self.agent.classify(x['user']['id'], x['user']['statuses_count'], x['user']['followers_count'],
                                     x['user']['friends_count'], x['user']['favourites_count'], x['user']['listed_count'])

        display_message(self.agent.aid.name, 'class of ' +
                        str(x['user']['id']) + ' is ' + str(classy))

        message = ACLMessage(ACLMessage.INFORM)
        message.set_protocol(ACLMessage.FIPA_REQUEST_PROTOCOL)
        message.add_receiver(agent_referee.aid)
        message.set_content(str(x['user']['id']) +
                            '¢' + str(classy[0]) + '¢' + str(classy[1]) + '¢' + str(x['created_at']))
        self.agent.send(message)


class AgentClassifier(Agent):

    def __init__(self, aid, ML_model):
        super(AgentClassifier, self).__init__(aid)
        self.call_later(8.0, self.launch_subscriber_protocol)
        self.ML_model = ML_model

    def launch_subscriber_protocol(self):
        msg = ACLMessage(ACLMessage.SUBSCRIBE)
        msg.set_protocol(ACLMessage.FIPA_SUBSCRIBE_PROTOCOL)
        msg.set_content('Subscription request')
        msg.add_receiver(agent_colector.aid)

        self.protocol = SubscriberProtocolClassifier(self, msg)
        self.behaviours.append(self.protocol)
        self.protocol.on_start()

    def classify(self, id, statuses_count, followers_count, friends_count, favourites_count, listed_count):
        if self.ML_model != None:
            d = {
                "statuses_count": [statuses_count],
                "followers_count": [followers_count],
                "friends_count": [friends_count],
                "favourites_count": [favourites_count],
                "listed_count": [listed_count],
            }

            df = pd.DataFrame(data=d)
            prediction = self.ML_model.predict(df)
            prediction_proba = self.ML_model.predict_proba(df)
            return [prediction[0], prediction_proba[0][0]]
        else:
            return random.randrange(2)


if __name__ == '__main__':

    number_of_classifiers = 3

    agents = list()
    port = int(argv[1])
    k = 100

    # setup the classifiers
    modelRFSoc1 = pickle.load(open('classifiers/modelRFTrad', 'rb'))
    modelSVMSoc1 = pickle.load(open('classifiers/modelSVMTrad', 'rb'))
    modelNBSoc1 = pickle.load(open('classifiers/modelNBTrad', 'rb'))

    participants = list()
    # defines the colector
    agent_name = 'agent_colector_{}@localhost:{}'.format(port, port)
    participants.append(agent_name)
    agent_colector = AgentColector(
        AID(name=agent_name), number_of_classifiers, offline=False)
    agents.append(agent_colector)

    # define the classifiers
    agent_name = 'agent_classifier_RF@localhost:{}'.format(port + k)
    participants.append(agent_name)
    agent_classifier_1 = AgentClassifier(AID(name=agent_name), modelRFSoc1)
    agents.append(agent_classifier_1)
    agent_name = 'agent_classifier_SVM@localhost:{}'.format(port - k)
    agent_classifier_2 = AgentClassifier(AID(name=agent_name), modelSVMSoc1)
    participants.append(agent_name)
    agents.append(agent_classifier_2)
    agent_name = 'agent_classifier_NB@localhost:{}'.format(port + k + k)
    agent_classifier_3 = AgentClassifier(AID(name=agent_name), modelNBSoc1)
    participants.append(agent_name)
    agents.append(agent_classifier_3)

    # defines the referee
    agent_name = 'agent_referee_{}@localhost:{}'.format(
        port + k + k + k, port + k + k + k)
    agent_referee = AgentReferee(
        AID(name=agent_name), participants, number_of_classifiers)
    participants.append(agent_name)
    agents.append(agent_referee)

    start_loop(agents)
