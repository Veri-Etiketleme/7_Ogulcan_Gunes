#!coding=utf-8
from pade.misc.utility import display_message
from pade.core.agent import Agent
from pade.acl.messages import ACLMessage, AID
# from pade.acl.aid import AID


# #!coding=utf-8
# from pade.misc.utility import display_message
# from pade.core.agent import Agent
# from pade.acl.aid import AID



# class ClassifierAgent(Agent):
     
#     def __init__(self, aid):
#         super(ClassifierAgent, self).__init__(aid=aid)
#         display_message(self.aid.localname, 'Hello World!')

#     def react(self, message):
#         display_message(self.aid.localname, 'Mensagem recebida')
#         display_message(self.aid.localname, message)

class ColectorAgent(Agent):

    def __init__(self, aid):
        super(ColectorAgent, self).__init__(aid=aid)
        display_message(self.aid.localname, 'Hello World!')
        
    def on_start(self):
        display_message(self.aid.localname, 'Enviando Mensagem')
        message = ACLMessage(ACLMessage.INFORM)
        message.add_receiver(AID('classifier1'))
        message.set_content('Ola')
        self.send(message)


    # def on_start(self):
    #     display_message(self.aid.localname, 'Enviando Mensagem')
    #     message = ACLMessage(ACLMessage.INFORM)
    #     message.add_receiver(AID('classifier_20001@localhost:20001'))
    #     message.set_content('Ola')

    def react(self, message):
        pass

    
