import numpy as np

def fair_sharer(values, num_iterations, share=0.1):
    values_new = np.array(values)
    for i in range(num_iterations):
        max_ind = np.argmax(values_new)
        max_val = values_new[max_ind]
        values_new[max_ind] -= max_val * share * 2
        values_new[max_ind - 1] += max_val * share
        if not max_ind + 1 > values_new.shape[0]:
            values_new[max_ind + 1] += max_val * share
            continue
        values_new[0] += max_val * share
    return values_new
