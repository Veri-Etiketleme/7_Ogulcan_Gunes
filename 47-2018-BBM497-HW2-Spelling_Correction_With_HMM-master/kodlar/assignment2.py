import sys
import re
import time

import constants
import NLP_v2
import NLP
import utils

# ins_del_subs_hash_pickle = utils.getArrayFromPickle(constants.INS_DEL_SUBS_PICKLE)
# misspell_chars_hash_pickle = utils.getArrayFromPickle(constants.MISSPELL_CHAR_HASH_PICKLE)
#
# misspelling_chars_hash = misspell_chars_hash_pickle
# ins_del_subs_hash = ins_del_subs_hash_pickle

ins_del_subs_hash = {}  # {wach:[each,watch,wac]}
misspelling_chars_hash = {}  # {(w,e):3, (at,a):1, (ch,c):4}

errorless_data = []  # misspell corrected data
wrong_correction_hash = {}  # {siter:[sister]}


def populateErrorlessLine(line):
    # is there any </ERR> tag
    if utils.containsError(line):
        # get errors
        error_list = re.findall(constants.ERROR_REGEX, line)
        for error in error_list:
            # get correction and misspell word from error tag
            correct = re.findall(constants.CORRECT_WORD_REGEX, error)[0].strip()
            wrong = re.findall(constants.WRONG_WORD_REGEX, error)[0].strip()

            # replace with correct one
            replace_regex = r'<ERR\s*targ=\s*' + correct + r'.*?</ERR>'
            line = re.sub(replace_regex, correct, line)

            utils.add_to_dict_as_array(wrong_correction_hash, wrong.lower(), correct.lower())

    line = line.lower().strip("\n").strip()
    # is line empty
    if line:
        errorless_data.append(line)


def readInputFile(INPUT_FILE):
    with open(INPUT_FILE, "r") as file:
        for line in file:
            populateErrorlessLine(line)
    utils.writeLog("DATA-PARSE", "All ERROR tags are parsed successfully.")


def calculateMisspelling(unigram_model):
    utils.writeLog("MISSPELLING",
                   "Misspelling (INSERTION-DELETION-SUBSTITUTION) calculation is started.")
    for wrong in wrong_correction_hash.keys():  # all of error words
        len_wrong = len(wrong)
        for word in unigram_model.getModelFrequency().keys():  # all unique keys of all data
            if abs(len(word) - len_wrong) <= 1:  # avoid too many process
                edit_distance = NLP.MinimumEditDistance(correct=word, wrong=wrong).calculate()
                transformation_type = edit_distance.getCorrectionType()
                # index = edit_distance.getMisspellIndex()

                if edit_distance.getEditDistance() == 1:
                    utils.add_to_dict_as_count(misspelling_chars_hash, edit_distance.getEditChars())

                    if transformation_type != NLP_v2.MID_TYPE_NONE:
                        utils.add_to_dict_as_array(ins_del_subs_hash, wrong, word)
                    else:
                        # check edit distance problems (It never happened, but it's good to check.)
                        utils.writeLog("WARNING",
                                       "Correct= " + word + " Wrong= " + wrong + "->  Min Edit distance is " + str(
                                           edit_distance.getEditDistance()))
    # utils.writeAsSerializable(ins_del_subs_hash, constants.INS_DEL_SUBS_PICKLE)
    # utils.writeAsSerializable(misspelling_chars_hash, constants.MISSPELL_CHAR_HASH_PICKLE)
    utils.writeLog("MISSPELLING",
                   "Misspelling (INSERTION-DELETION-SUBSTITUTION) calculation is completed for every ERROR word.")


def calculateHMM(unigram_model, bigram_model, INPUT_FILE, OUT_FILE):
    utils.writeLog("Hidden-Markov-Model", "Hidden Markov Model with Viterbi Algorithm has started.")
    output_file = open(OUT_FILE, "w")
    total_found = 0
    total_error = 0
    with open(INPUT_FILE, "r") as file:
        for line in file:
            line = line.strip("\n").strip()
            if line:
                if constants.HMM_VERSION == 1:
                    founded, total = NLP.HiddenMarkovModel(line, unigram_model, bigram_model,
                                                           misspelling_chars_hash, ins_del_subs_hash, output_file) \
                        .generateHmmModel() \
                        .getFoundPerTotal()
                else:
                    founded, total = NLP_v2.HiddenMarkovModel(line, unigram_model, bigram_model,
                                                              misspelling_chars_hash, ins_del_subs_hash, output_file) \
                        .generateHmmModel() \
                        .getFoundPerTotal()
                total_found += founded
                total_error += total
    result = round((total_found / total_error) * 100)
    utils.writeEvaluationBoxToFile(output_file, result)
    output_file.close()
    utils.writeLog("Hidden-Markov-Model", "Hidden Markov Model with Viterbi Algorithm finished for each sentences.")


def main(argv):
    INPUT_FILE = argv[1]
    OUT_FILE = argv[2]

    readInputFile(INPUT_FILE=INPUT_FILE)

    unigram_model = NLP.NGramModel(errorless_data, NLP_v2.UNIGRAM).createModel()
    bigram_model = NLP.NGramModel(errorless_data, NLP_v2.BIGRAM).createModel()

    calculateMisspelling(unigram_model)
    calculateHMM(unigram_model, bigram_model, INPUT_FILE, OUT_FILE)


if __name__ == "__main__":
    utils.writeLog("MAIN-PROCESS", "Spelling Correction program is started.")
    utils.writeLog("VERSION", "You are using Hidden Markov Model Version: " + str(constants.HMM_VERSION))
    start = time.time()
    main(sys.argv)
    utils.writeLog("MAIN-PROCESS",
                   "Spelling Correction program is finished in " + str(time.time() - start) + " seconds.")
