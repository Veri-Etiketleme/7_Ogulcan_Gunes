ERROR_END_TAG = "</ERR>"
ERROR_REGEX = r'<ERR.*?<\/ERR>'
ERROR_REGEX_WITH_DELIMITER = "(<ERR.*?<\/ERR>)"
CORRECT_WORD_REGEX = r'(?<=targ=).+?(?=>)'
WRONG_WORD_REGEX = r'(?<=>).+?(?=</ERR>)'
ANY_WHITESPACE_REGEX = r'\s+'

INS_DEL_SUBS_PICKLE = "ins-del-subs.bin"  # to instant run, i will not use original assignment code
MISSPELL_CHAR_HASH_PICKLE = "misspell-char-hash.bin"

LOG_FILE = "console.log"

HMM_VERSION = 1
