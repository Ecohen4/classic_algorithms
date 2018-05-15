import numpy as np
from src.sort_list import bubble_sort

def binary_search(list_, item, verbose=False):
    print ("searching for {} in list...".format(item))
    found, i, n = False, 0, len(list_)
    max_iter = np.ceil(np.log2(n))
    start, end = 0, n-1
    list_ = bubble_sort(list_)
    while (not found) and (i <= max_iter):
        i += 1
        mid = (start + end) // 2
        if item == list_[mid]:
            found = True
        elif item < list_[mid]:
            end = mid
        elif item > list_[mid]:
            start = mid
        if verbose:
            print ("searching {branch}?".format(branch=list_[start:end]))
            print ("    is {item} == {value}?".format(item=item, value=list_[mid]))
            print("    {found}".format(found=found))
    print("found" if found else "not found")
    return found


if __name__ == "__main__":
    import random
    l = random.choices(population=range(1000), k=1000)
    i = random.randint(0, 1000)
    binary_search(list_=sorted(l), item=i)
