#!coding=utf-8
from pade.behaviours.protocols import FipaSubscribeProtocol
from pade.misc.utility import display_message
from pade.acl.messages import ACLMessage
from pade.core.agent import Agent
import random
import ast

class SubscriberProtocolClassifier(FipaSubscribeProtocol):

    def __init__(self, agent, message):
        super(SubscriberProtocolClassifier, self).__init__(
            agent, message, is_initiator=True)

    def handle_agree(self, message):
        display_message(self.agent.aid.name, message.content)

    def handle_inform(self, message):
        x = ast.literal_eval(str(message.content))
        classy = self.agent.classify(x['user']['id'], x['user']['statuses_count'], x['user']['followers_count'], x['user']['friends_count'], x['user']['favourites_count'], x['user']['listed_count'])
        
        display_message(self.agent.aid.name, 'class of ' +
                        str(x['user']['id']) + ' is ' + str(classy))
        from pade.acl.messages import ACLMessage, AID

        message = ACLMessage(ACLMessage.INFORM)
        message.set_protocol(ACLMessage.FIPA_REQUEST_PROTOCOL)
        message.add_receiver(agent_referee.aid)
        message.set_content(str(x['user']['id']) + ' ' + str(classy))
        self.agent.send(message)

class AgentClassifier(Agent):

    def __init__(self, aid, colector_aid):
        super(AgentClassifier, self).__init__(aid)
        self.call_later(8.0, self.launch_subscriber_protocol)
        self.colector_aid = colector_aid

    def launch_subscriber_protocol(self):
        msg = ACLMessage(ACLMessage.SUBSCRIBE)
        msg.set_protocol(ACLMessage.FIPA_SUBSCRIBE_PROTOCOL)
        msg.set_content('Subscription request')
        msg.add_receiver(self.colector_aid)

        self.protocol = SubscriberProtocolClassifier(self, msg)
        self.behaviours.append(self.protocol)
        self.protocol.on_start()

    def classify(self, id, statuses_count, followers_count, friends_count, favourites_count, listed_count):
        print(id, statuses_count, followers_count, friends_count, favourites_count, listed_count)
        return random.randrange(2)
