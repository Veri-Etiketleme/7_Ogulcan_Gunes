import string
import numpy as np
import re

import NLP
import constants
import utils

MID_TYPE_INSERTION = "insertion"
MID_TYPE_DELETION = "deletion"
MID_TYPE_SUBTITUTION = "substitution"
MID_TYPE_NONE = "none"

UNIGRAM = 1
BIGRAM = 2  # Default Constants
TRIGRAM = 3


class HiddenMarkovModel:
    __error_line = None
    __unigram_model = None
    __bigram_model = None
    __misspell_chars = None
    __ins_del_subs_hash = None

    __found = None
    __total = None
    __possible_sentence = None

    __output_file = None

    __transition_frequency = {}

    def __init__(self, error_line, unigram_model, bigram_model, misspell_chars, ins_del_subs_hash, output_file):
        self.__error_line = error_line
        self.__unigram_model = unigram_model
        self.__bigram_model = bigram_model
        self.__misspell_chars = misspell_chars
        self.__ins_del_subs_hash = ins_del_subs_hash
        self.__found = 0
        self.__total = 0
        self.__possible_sentence = []
        self.__transition_frequency = {}
        self.__output_file = output_file

    def getPossibleSentence(self):
        return self.__possible_sentence

    def getFoundPerTotal(self):
        return self.__found, self.__total

    def generateHmmModel(self):
        self.__error_line = self.__error_line.strip()
        # creates word array from sentence with error like ["i", "<ERR targ=Ferrari> Farrari </ERR>", "and"] etc.
        sentence_like_list = self.__constructSentenceLikeList(self.__error_line, True)
        # traverse two pairs
        repeat = len(sentence_like_list) - BIGRAM + 1

        # initial viterbi lists
        subsentence_list = [[]]
        probability_list = [1]

        for i in range(repeat):
            first = sentence_like_list[i]
            second = sentence_like_list[i + 1]

            if utils.containsError(first):

                if utils.containsError(second):

                    ####        ERROR - ERROR ------>   START ####

                    second_wrong = re.findall(constants.WRONG_WORD_REGEX, second)[0].strip()

                    if second_wrong in self.__ins_del_subs_hash:
                        # get all transitions
                        check_list = self.__ins_del_subs_hash[second_wrong]
                        temp_sub = []
                        temp_prob = []
                        for transition in check_list:
                            temp_prob_inner = []
                            for sub, prob in zip(subsentence_list, probability_list):
                                last_word = utils.getLastWordFromSentenceList(sub)
                                # calculate probability of transition
                                temp_prob_inner.append(prob * self.__bigram_model.getBigramProbabilityFor(
                                    self.__unigram_model.getModelFrequency(),
                                    last_word,
                                    transition
                                ) * self.__getMisspellProbabilityFor(second_wrong, transition))
                            # get max probability of transition
                            index = np.argmax(temp_prob_inner)
                            temp_prob.append(probability_list[index] * temp_prob_inner[index])
                            subsentence_list_temp = subsentence_list[index][:]
                            subsentence_list_temp.append(transition)
                            temp_sub.append(subsentence_list_temp)
                        # update initial lists
                        subsentence_list = temp_sub
                        probability_list = temp_prob
                    else:
                        # there is no transitions for word
                        temp_prob = []
                        unigram_freqs = []
                        for sub_list, prob in zip(subsentence_list, probability_list):
                            last_word = utils.getLastWordFromSentenceList(sub_list)
                            if last_word in self.__unigram_model.getModelFrequency():
                                unigram_freqs.append(self.__unigram_model.getModelFrequency()[last_word])
                            else:
                                unigram_freqs.append(0)
                            temp_prob.append(prob * self.__bigram_model.getBigramProbabilityFor(
                                self.__unigram_model.getModelFrequency(),
                                last_word,
                                second_wrong
                            ))

                        index = np.argmax(temp_prob)
                        if temp_prob[index] == 0:
                            index = np.argmax(unigram_freqs)
                        subsentence_list_temp = [subsentence_list[index][:]]
                        subsentence_list_temp[0].append(second_wrong)
                        subsentence_list = subsentence_list_temp
                        probability_list = [probability_list[index] * temp_prob[index]]

                    ####        ERROR - ERROR ------>   END ####

                else:

                    ####        ERROR - NOT ------>   START ####

                    # many to one probability
                    temp_prob = []
                    unigram_freqs = []
                    for sub_list, prob in zip(subsentence_list, probability_list):
                        last_word = utils.getLastWordFromSentenceList(sub_list)
                        if last_word in self.__unigram_model.getModelFrequency():
                            unigram_freqs.append(self.__unigram_model.getModelFrequency()[last_word])
                        else:
                            unigram_freqs.append(0)
                        temp_prob.append(prob * self.__bigram_model.getBigramProbabilityFor(
                            self.__unigram_model.getModelFrequency(),
                            last_word,
                            second
                        ))

                    # get max probability of transitions
                    index = np.argmax(temp_prob)
                    if temp_prob[index] == 0:
                        index = np.argmax(unigram_freqs)
                    subsentence_list_temp = [subsentence_list[index][:]]
                    subsentence_list_temp[0].append(second)
                    subsentence_list = subsentence_list_temp
                    probability_list = [probability_list[index] * temp_prob[index]]

                    ####        ERROR - NOT ------>   END ####

            else:
                if utils.containsError(second):

                    ####        NOT - ERROR ------>   START ####

                    # one to many probability
                    second_wrong = re.findall(constants.WRONG_WORD_REGEX, second)[0].strip()

                    if second_wrong in self.__ins_del_subs_hash:
                        # get transitions
                        check_list = self.__ins_del_subs_hash[second_wrong]
                        temp_sub = []
                        temp_prob = []
                        for iter in range(len(check_list)):
                            transition = check_list[iter]
                            temp_sub.append(subsentence_list[0][:])
                            temp_sub[iter].append(transition)
                            # calculate probability according to transitions
                            temp_prob.append(probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                                self.__unigram_model.getModelFrequency(),
                                first,
                                transition
                            ) * self.__getMisspellProbabilityFor(second_wrong, transition))
                        # update initial lists
                        subsentence_list = temp_sub
                        probability_list = temp_prob

                    else:
                        # there is not transitions, so one to one probability comes.
                        subsentence_list[0].append(second_wrong)
                        probability_list = [probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                            self.__unigram_model.getModelFrequency(),
                            first, second_wrong)]

                    ####        NOT - ERROR ------>   END ####

                else:

                    ####        NOT - NOT ------>   START ####

                    # one to one probability
                    subsentence_list[0].append(second)
                    # calculate bigram probability only
                    probability_list[0] = probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                        self.__unigram_model.getModelFrequency(),
                        first, second)

                    ####        NOT - NOT ------>   END ####

        self.__total = self.__getErrorCount()

        sentence = " ".join(subsentence_list[0][:len(subsentence_list[0]) - 1]).strip().strip(
            string.punctuation).strip()

        # write generated sentences to output file

        utils.writeLineToFile(self.__output_file, sentence)

        # evaluate sentence

        sentence = re.split(constants.ANY_WHITESPACE_REGEX, sentence)
        correct_sentence = self.__constructSentenceLikeList(self.__error_line, False)

        for iter in range(len(correct_sentence)):
            # align two sentence to evaluation
            word = correct_sentence[iter]
            if constants.ERROR_END_TAG in word:
                correct = re.findall(constants.CORRECT_WORD_REGEX, word)[0].strip()
                if correct == sentence[iter]:
                    self.__found += 1
                else:
                    targ_words = re.split(constants.ANY_WHITESPACE_REGEX, correct)
                    if len(targ_words) > 1:
                        temp = True
                        for i in range(len(targ_words)):
                            if targ_words[i] != sentence[iter + i]:
                                temp = False
                                break
                        if temp:
                            self.__found += 1

        return self

    def __getErrorCount(self):
        return len(re.findall(constants.ERROR_REGEX, self.__error_line))

    def __getErrorlessLine(self, line):
        # is there any </ERR> tag
        if utils.containsError(line):
            # get errors
            error_list = re.findall(constants.ERROR_REGEX, line)
            for error in error_list:
                # get correction and misspell word from error tag
                correct = re.findall(constants.CORRECT_WORD_REGEX, error)[0].strip()

                # replace with correct one
                replace_regex = r'<ERR\s*targ=\s*' + correct + r'.*?</ERR>'
                line = re.sub(replace_regex, correct, line)

        line = line.lower().strip("\n").strip().strip(string.punctuation).strip()
        # is line empty
        if line:
            return line
        else:
            return ""

    def __constructSentenceLikeList(self, error_sentence, putWordBoundry):
        splitted_list = []
        temp_list = re.split(constants.ERROR_REGEX_WITH_DELIMITER, error_sentence)
        for element in temp_list:
            if utils.containsError(element):
                element = element.strip()
                if element:
                    splitted_list.append(element)
            else:
                element = element.strip()
                sub_element = re.split(constants.ANY_WHITESPACE_REGEX, element)
                for elem in sub_element:
                    elem = elem.strip(string.punctuation).strip()
                    if elem:
                        splitted_list.append(elem.lower())

        if putWordBoundry:
            splitted_list.insert(0, "<s>")
            splitted_list.append("</s>")

        return splitted_list

    def __getNumOfStringInAllWords(self, query):
        if query in self.__transition_frequency:
            return self.__transition_frequency[query]
        else:
            count = 0
            for word in self.__unigram_model.getWordList():
                if query in word:
                    count += 1
            self.__transition_frequency[query] = count
            return count

    def __getMisspellProbabilityFor(self, wrong, correct):
        min_edit_distance = NLP.MinimumEditDistance(correct, wrong).calculate()
        transition = min_edit_distance.getEditChars()
        correction_type = min_edit_distance.getCorrectionType()
        probability = 0
        if transition in self.__misspell_chars:
            if correction_type != MID_TYPE_NONE:
                probability = self.__misspell_chars[transition] / self.__getNumOfStringInAllWords(transition[0])
        return probability
