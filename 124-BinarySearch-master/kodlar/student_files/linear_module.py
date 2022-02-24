

""" Linear/sequential searching """
import tools
# uncomment the next line if you want to make some Name objects
from classes import Name
import time
start_time = time.time()

"""linear search programme developed by Okoko Anainga 22/08/2020"""
def linear_result_finder(tested_list, quarantined):
    """ The tested list contains (nhi, Name, result) t/uples
        and isn't guaranteed to be in any order
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
    """
    
    comparisons = 0
    results = []
    for quartined_name in quarantined:
        if len(tested_list) == 0:
            tup = (quartined_name, None, None)
            results.append(tup) 
        else:
            found = False
            counter = 0
            
            while counter < len(tested_list) and not found:
                comparisons += 1
                nhi, name, result = tested_list[counter]
                if name == quartined_name:
                    tup = (name, nhi, result)
                    results.append(tup)
                    found = True
                else:
                    counter += 1
                if counter == len(tested_list) and not found:
                    tup = (quartined_name, None, None)
                    results.append(tup)
                    found = True
    
    return results, comparisons


# Don't submit your code below or pylint will get annoyed :)
if __name__ == '__main__':
    # write your own simple tests here
    # eg
    from linear_module import linear_result_finder
    filename = "test_data/test_data-10n-10r-1-a.txt"   
    tested, quarantined, expected_results = tools.read_test_data(filename)
    print(linear_result_finder(tested, quarantined))
    
    print("Time for run %s in seconds" % (time.time()-start_time))
    
