def insertion_sort(input_list):
    n = len(input_list)
    for i in range(1,n):
        val = input_list[i]
        while i > 0 and input_list[i-1] > val:
            input_list[i] = input_list[i-1]
            i -= 1
        input_list[i] = val
    return input_list


def bubble_sort(input_list):
    n = len(input_list)
    for i in range(n-1, 1, -1):
        for j in range(i):
            if input_list[j] > input_list[j+1]:
                larger_val = input_list[j]
                smaller_val = input_list[j+1]
                input_list[j] = smaller_val
                input_list[j+1] = larger_val
    return input_list
