

import re
import datetime


def digits(text):
    """
    :param text: input test (Str)
    :return: text with only digits (Str)

    Return string with only digits.
    """
    return re.sub("\D", "", text)


def stredup(mylist):
    """
    :param mylist: List with elements (List)
    :return: List with elements with no duplicates (List)

    Removes duplicates from list with string version of elements.
    """
    return redup([str(e) for e in mylist])


def redup(mylist):
    """
    :param mylist: List with elements (List)
    :return: List with elements with no duplicates (List)

    Removes duplicates from list.
    """
    return list(dict.fromkeys(mylist))


def with_quotes(text):
    """
    :param text: Text (Str)
    :return: Text where every list of non-special characters is in quotes (Str)

    Adds quotes around every chain of non-special characters.
    """
    labels = re.findall(r"\W+", text)
    for label in redup(labels):
        text = text.replace(label, '"'+label+'"')
    return text.replace('"."', '.')


def l2t(l, delimiter=' ', minimal=0):
    """
    :param l: List with strings (List)
    :param delimiter: Delimiter between words (Str)
    :param minimal: minimal length of word to be count in concatenated string (Int)
    :return: Concatenated and delimited by delimiter string (Str)

    Joins strings from list using specified delimiter.
    """
    if type(l) != list:
        return str(l)
    return delimiter.join(list(filter(lambda x: len(x) > minimal, l)))


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def subsets(length):
    """
    :param length: Int
    :return: List of List of Int

    Takes length of a needed subsets and return list with permutations of (-1, 1) for a given length.
    """
    results = [[]]
    for l in range(length):
        res = results[:]
        a = [lis[:] + [1] for lis in res]
        b = [lis[:] + [-1] for lis in res]

        results.extend(a)
        results.extend(b)
    return [r for r in results if len(r) == length]


def now():
    """
    :return: Time now (Str)

    Returns string with time in format YYYY-MM-DD_HH-MM-SS, used mainly with creating file names.
    """
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')