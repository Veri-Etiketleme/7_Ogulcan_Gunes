import string
import numpy as np
import re

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
        sentence_like_list = self.__constructSentenceLikeList(self.__error_line)
        repeat = len(sentence_like_list) - BIGRAM + 1

        subsentence_list = [""]
        probability_list = [1]

        for i in range(repeat):
            first = sentence_like_list[i]
            second = sentence_like_list[i + 1]

            if utils.containsError(first):

                if utils.containsError(second):

                    first_wrong = re.findall(constants.WRONG_WORD_REGEX, first)[0].strip()
                    first_correct = re.findall(constants.CORRECT_WORD_REGEX, first)[0].strip()

                    second_wrong = re.findall(constants.WRONG_WORD_REGEX, second)[0].strip()
                    second_correct = re.findall(constants.CORRECT_WORD_REGEX, second)[0].strip()

                    if second_wrong in self.__ins_del_subs_hash:
                        check_list = self.__ins_del_subs_hash[second_wrong]
                        temp_sub = []
                        temp_prob = []
                        for transition in check_list:
                            temp_prob_inner = []
                            for sub, prob in zip(subsentence_list, probability_list):
                                last_word = utils.getLastWordFromSentence(sub)
                                temp_prob_inner.append(prob * self.__bigram_model.getBigramProbabilityFor(
                                    self.__unigram_model.getModelFrequency(),
                                    last_word,
                                    transition
                                ) * self.__getMisspellProbabilityFor(second_wrong, transition))
                            index = np.argmax(temp_prob_inner)
                            temp_prob.append(probability_list[index] * temp_prob_inner[index])
                            temp_sub.append(subsentence_list[index] + transition + " ")
                        subsentence_list = temp_sub
                        probability_list = temp_prob
                    else:
                        temp_prob = []
                        for sub, prob in zip(subsentence_list, probability_list):
                            last_word = utils.getLastWordFromSentence(sub)
                            temp_prob.append(prob * self.__bigram_model.getBigramProbabilityFor(
                                self.__unigram_model.getModelFrequency(),
                                last_word,
                                second_wrong
                            ))

                        index = np.argmax(temp_prob)
                        subsentence_list = [subsentence_list[index] + second_wrong + " "]
                        probability_list = [probability_list[index] * temp_prob[index]]

                else:
                    temp_prob = []
                    for sub, prob in zip(subsentence_list, probability_list):
                        last_word = utils.getLastWordFromSentence(sub)
                        temp_prob.append(prob * self.__bigram_model.getBigramProbabilityFor(
                            self.__unigram_model.getModelFrequency(),
                            last_word,
                            second
                        ))

                    index = np.argmax(temp_prob)
                    subsentence_list = [subsentence_list[index] + second + " "]
                    probability_list = [probability_list[index] * temp_prob[index]]

            else:
                if utils.containsError(second):
                    second_wrong = re.findall(constants.WRONG_WORD_REGEX, second)[0].strip()
                    second_correct = re.findall(constants.CORRECT_WORD_REGEX, second)[0].strip()
                    if second_wrong in self.__ins_del_subs_hash:
                        check_list = self.__ins_del_subs_hash[second_wrong]
                        temp_sub = []
                        temp_prob = []
                        for transition in check_list:
                            temp_sub.append(subsentence_list[0] + transition + " ")
                            temp_prob.append(probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                                self.__unigram_model.getModelFrequency(),
                                first,
                                transition
                            ) * self.__getMisspellProbabilityFor(second_wrong, transition))
                        subsentence_list = temp_sub
                        probability_list = temp_prob

                    else:
                        subsentence_list = [subsentence_list[0] + second_wrong + " "]
                        probability_list = [probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                            self.__unigram_model.getModelFrequency(),
                            first, second_wrong)]
                else:
                    subsentence_list = [subsentence_list[0] + second + " "]
                    probability_list = [probability_list[0] * self.__bigram_model.getBigramProbabilityFor(
                        self.__unigram_model.getModelFrequency(),
                        first, second)]

        sentence = re.split(constants.ANY_WHITESPACE_REGEX, subsentence_list[0].strip())
        sentence = " ".join(sentence[0:len(sentence) - 1]).strip()
        if sentence:
            utils.writeLineToFile(self.__output_file, sentence)
            total_error_count = len(re.findall(constants.ERROR_REGEX, self.__error_line))
            sentence = re.split(constants.ANY_WHITESPACE_REGEX, sentence)
            correct_sentence = re.split(constants.ANY_WHITESPACE_REGEX,
                                        self.__getErrorlessLine(self.__error_line).strip())

            not_found = 0
            if len(sentence) == len(correct_sentence):
                for i in range(len(sentence)):
                    if sentence[i] != correct_sentence[i]:
                        not_found += 1
            self.__total = total_error_count
            self.__found = total_error_count - not_found

        return self

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

        line = line.lower().strip("\n").strip().strip(string.punctuation)
        # is line empty
        if line:
            return line
        else:
            return ""

    def __constructSentenceLikeList(self, error_sentence):
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
        min_edit_distance = MinimumEditDistance(correct, wrong).calculate()
        transition = min_edit_distance.getEditChars()
        correction_type = min_edit_distance.getCorrectionType()
        probability = 0
        if transition in self.__misspell_chars:
            if correction_type != MID_TYPE_NONE:
                probability = self.__misspell_chars[transition] / self.__getNumOfStringInAllWords(transition[0])
        return probability


"""

Creates minimum edit distance dynamic matrix and calculates edit distance.
If edit distance is 1, checks misspell correction type (INSERTION, DELETION,SUBSTITUTION)
Example: "ebubekir" and "ebubekirr" -> MID_TYPE_INSERTION at index=8

"""


class MinimumEditDistance:
    __subs_cost = 1
    __correct = ""
    __wrong = ""
    __edit_matrix = []

    __edit_distance = -1
    __correction_type = MID_TYPE_NONE
    __misspell_index = -1
    __edit_chars = ()

    def __init__(self, correct, wrong):
        self.__correct = correct
        self.__wrong = wrong
        self.__edit_matrix = []
        self.__edit_distance = -1
        self.__subs_cost = 1
        self.__correction_type = MID_TYPE_NONE
        self.__misspell_index = -1
        self.__edit_chars = ()

    def setSubstitutionCost(self, cost=1):
        self.__subs_cost = cost
        return self

    def getDistanceMatrix(self):
        return self.__edit_matrix

    def getEditDistance(self):
        return self.__edit_distance

    def getMisspellIndex(self):
        return self.__misspell_index

    def getCorrectionType(self):
        return self.__correction_type

    def getEditChars(self):
        return self.__edit_chars

    def __populateMatrix(self):
        # initialize MED(Minimum Edit Distance) matrix
        correct_len = len(self.__correct)
        wrong_len = len(self.__wrong)
        self.__edit_matrix = np.zeros((correct_len + 1, wrong_len + 1), dtype=int)

        self.__edit_matrix[0] = [x for x in range(wrong_len + 1)]
        for i in range(correct_len + 1):
            self.__edit_matrix[i, 0] = i

    def calculate(self):
        correct_len = len(self.__correct)
        wrong_len = len(self.__wrong)

        self.__populateMatrix()
        for i in range(1, correct_len + 1):
            for j in range(1, wrong_len + 1):
                left = self.__edit_matrix[i][j - 1] + 1
                top = self.__edit_matrix[i - 1][j] + 1
                diag = self.__edit_matrix[i - 1][j - 1]

                if self.__correct[i - 1] != self.__wrong[j - 1]:
                    diag += self.__subs_cost

                min_dist = min((left, top, diag))  # np.min() 3 times slower than min()
                self.__edit_matrix[i, j] = min_dist

        # print(self.__edit_matrix)

        # update edit distance
        self.__edit_distance = self.__edit_matrix[correct_len][wrong_len]
        if self.__edit_distance == 1:
            self.__backtrace_and_compute_alignments()
        return self

    def __backtrace_and_compute_alignments(self):
        i = len(self.__correct)
        j = len(self.__wrong)
        while True:
            if i - 1 < 0:
                # insertion at beginning of word
                self.__correction_type = MID_TYPE_INSERTION
                self.__misspell_index = 0
                self.__edit_chars = (self.__correct[0], self.__wrong[:2])
                break
            if j - 1 < 0:
                # deletion at beginning of word
                self.__correction_type = MID_TYPE_DELETION
                self.__misspell_index = 0
                self.__edit_chars = (self.__correct[:2], self.__wrong[0])
                break

            left = self.__edit_matrix[i][j - 1] + 1
            top = self.__edit_matrix[i - 1][j] + 1
            diag = self.__edit_matrix[i - 1][j - 1]

            subs_flag = False

            if self.__correct[i - 1] != self.__wrong[j - 1]:
                diag += self.__subs_cost
                subs_flag = True

            min_dist = np.argmin((diag, left, top))
            if min_dist == 0:
                if subs_flag:
                    self.__correction_type = MID_TYPE_SUBTITUTION
                    self.__misspell_index = j - 1
                    self.__edit_chars = (self.__correct[i - 1], self.__wrong[j - 1])
                    break
                else:
                    i -= 1
                    j -= 1
            elif min_dist == 1:
                self.__correction_type = MID_TYPE_INSERTION
                self.__misspell_index = j - 1
                self.__edit_chars = (self.__correct[self.__misspell_index - 1],
                                     self.__wrong[self.__misspell_index - 1: self.__misspell_index + 1])
                break
            elif min_dist == 2:
                self.__correction_type = MID_TYPE_DELETION
                self.__misspell_index = i - 1
                self.__edit_chars = (self.__correct[self.__misspell_index - 1:self.__misspell_index + 1],
                                     self.__wrong[self.__misspell_index - 1])
                break
        # print(self.__misspell_index, self.__correction_type)


"""
Creates N-Gram language model
N-Gram default is UNIGRAM but you can use any number of positive integers
Example: Create 10-Gram Language Model

"""


class NGramModel:
    __N_GRAM = UNIGRAM

    __sentences = []
    __words = []
    __NGram_model = {}

    __sum_model_count = 0
    __len_model_count = 0

    def __init__(self, sentences, N_gram):
        self.__N_GRAM = N_gram
        self.__sentences = sentences
        self.__words = []
        self.__NGram_model = {}
        self.__sum_model_count = 0
        self.__len_model_count = 0

    # creates model with given constructor parameters
    def createModel(self):
        self.__populateWordList()
        return self

    def getModelFrequency(self):
        return self.__NGram_model

    def getWordList(self):
        return self.__words

    def getModelFrequencyLen(self):
        return self.__len_model_count

    def getModelFrequencySum(self):
        return self.__sum_model_count

    def getBigramProbabilityFor(self, unigram_freq, first, second):
        bigram_count = 0
        unigram_count = 0

        bigram_word = " ".join((first, second))

        if bigram_word in self.__NGram_model:
            bigram_count = self.__NGram_model[bigram_word]

        if first in unigram_freq:
            unigram_count = unigram_freq[first]

        if bigram_count == 0 or unigram_count == 0:
            return 0
        else:
            return bigram_count / unigram_count

    # gets a dictionary that contains keys that starts with key value
    # i couldn't use Default dict because i already finished my work :) sorry :)
    def __getStartsWith(self, key):
        temp = {}
        for i, j in self.__NGram_model.items():
            if i.startswith(key):
                temp[i] = j
        return temp

    # adds string boundaries to sentences according to n-grams
    def __setNgramStringBoundry(self, sentence, N_gram):
        if N_gram == UNIGRAM:
            sentence = sentence.strip()
            sentence = "<s> " + sentence + " </s>"
            return sentence
        else:
            n_gram = N_gram - 1
            sentence = "<s> " * n_gram + sentence + " </s>"
            return sentence

    # populates frequency dictionary
    def __populateWordList(self):
        for sentence in self.__sentences:
            words = re.compile(constants.ANY_WHITESPACE_REGEX).split(sentence)

            temp = []
            for word in words:
                word = word.strip().strip(string.punctuation).strip()
                if word:
                    temp.append(word)
            words = temp
            self.__words.extend(words)

            sentence = " ".join(words)
            sentence = self.__setNgramStringBoundry(sentence, self.__N_GRAM)
            # get all words with whitespace regex
            words = re.compile(constants.ANY_WHITESPACE_REGEX).split(sentence)

            repeat = len(words) - self.__N_GRAM + 1
            for i in range(repeat):
                n_set = words[i:i + self.__N_GRAM]
                result = " ".join(n_set).strip()
                if result:
                    if result in self.__NGram_model:
                        self.__NGram_model[result] += 1
                    else:
                        self.__NGram_model[result] = 1
        self.__len_model_count = len(self.__NGram_model)
        self.__sum_model_count = sum(self.__NGram_model.values())
        utils.writeLog("Language-Model", "Frequency Dictionary for " + str(
            self.__N_GRAM) + "-Gram is built successfully.")
