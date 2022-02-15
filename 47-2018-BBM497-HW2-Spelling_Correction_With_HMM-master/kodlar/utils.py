import pickle
import re
import time

import constants


def getLastWordFromSentence(sentence):
    sentence = sentence.strip()
    words_list = re.split(constants.ANY_WHITESPACE_REGEX, sentence)
    return words_list[len(words_list) - 1]


def getLastWordFromSentenceList(sentence):
    return sentence[len(sentence) - 1]


def isSentenceBoundry(string):
    return string == "<s>" or string == "</s>"


def containsError(string):
    return constants.ERROR_END_TAG in string


def add_to_dict_as_array(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


def add_to_dict_as_count(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def writeLineToFile(file, buffer):
    toWrite = buffer + "\n"
    file.write(toWrite)


def writeEvaluationBoxToFile(file, result):
    file.write("\n\n")
    buffer = "#     Evaluation Result: %" + str(result) + "     #"
    file.write("#" * len(buffer))
    file.write(buffer)
    file.write("#" * len(buffer))
    file.write("\n\n")


def writeLog(tag, log):
    f = open(constants.LOG_FILE, "a")
    buffer = time.strftime("%H:%M:%S") + "   LOG/ " + tag.upper() + ":  " + log + "\n"
    f.write(buffer)
    f.close()


def writeAsSerializable(array, fileName):
    file = open(fileName, 'wb')
    pickle.dump(array, file)
    file.close()
    writeLog("PICKLE", "PICKLE file is created as " + fileName)


def getArrayFromPickle(fileName):
    file = open(fileName, 'rb')
    array = pickle.load(file)
    # file.close()
    writeLog("PICKLE", "Array is get from pickle file " + fileName)
    return array
