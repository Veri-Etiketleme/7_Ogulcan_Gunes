#!coding=utf-8
from pade.misc.utility import start_loop
from sys import argv
from pade.acl.aid import AID
# Agents files
from colector import ColectorAgent
from classifier import AgentClassifier

if __name__ == '__main__':
    agents_per_process = 1
    # c = 0
    agents = list()

    colector_name = 'colector'
    colector = ColectorAgent(AID(name=colector_name))
    agents.append(colector)
    colector_aid = ''
    
    for i in range(agents_per_process):
        # port = int(argv[1]) + c
        classifier_name = 'classifier{}'.format(i)
        classifier = AgentClassifier(AID(name=classifier_name), colector_aid)
        agents.append(classifier)
   
        
        # c += 1000

    start_loop(agents)
