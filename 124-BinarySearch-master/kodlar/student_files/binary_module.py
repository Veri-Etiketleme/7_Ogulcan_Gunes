

import tools
# uncomment the next line if you want to make some Name objects
from classes import Name
import time
start_time = time.time()


"""two comparison Binary search developed by Okoko Anainga 22/08/2020"""

def binary_result_finder(tested, quarantined):
    """ The tested list contains (nhi, Name, result) tuples and
        will be sorted by Name
        quarantined is a list of Name objects
        and isn't guaranteed to be in any order
        This function should return a list of (Name, nhi, result)
        tuples and the number of comparisons made
        The result list must be in the same order
        as the  quarantined list.
        The nhi and result should both be set to None if
        the Name isn't found in tested_list
        You must keep track of all the comparisons
        made between Name objects.
        Your function must not alter the tested_list or
        the quarantined list in any way.
        Note: You shouldn't sort the tested_list, it is already sorted. Sorting it
        will use lots of extra comparisons!
    """
    total_comparisons = 0
    results = []
    for quartined_name in quarantined:
        if len(tested) == 0:
            tup = (quartined_name, None, None)
            results.append(tup)
        else:
            total_comparisons += 1
            first = 0
            last = len(tested) - 1
            found = False
            while first <= last and not found:
                midpoint = (first + last) // 2
                nhi, name, result = tested[midpoint]
                if quartined_name < name:
                    total_comparisons += 1
                    last = midpoint - 1
                elif quartined_name > name:
                    total_comparisons += 1
                    first = midpoint + 1
                else:
                    total_comparisons += 1
                    tup = (quartined_name, nhi, result)
                    results.append(tup)
                    found = True
            if first > last and not found:
                tup = (quartined_name, None, None)
                results.append(tup)
                found = True

    return results, total_comparisons


# Don't submit your code below or pylint will get annoyed :)
if __name__ == '__main__':
    # feel free to do some of your simple tests here
    # eg,
    from binary_module import binary_result_finder
    filename = "test_data/test_data-10n-10r-1-a.txt"
    tested, quarantined, expected_results = tools.read_test_data(filename)
    print(binary_result_finder(tested, quarantined))
    print("Time for run %s in seconds" % (time.time()-start_time))
