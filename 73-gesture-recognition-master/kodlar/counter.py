def max_freq(ans_list):
    ans_counter={}
    for ans in ans_list:
        if ans in ans_counter:
            ans_counter[ans] +=1
        else:
            ans_counter[ans]=1
    popular_num = sorted(ans_counter, key = ans_counter.get, reverse = True)
    top=popular_num[:1]
    return top[0]
