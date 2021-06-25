import copy
import re
import csv
import pickle
from collections import deque
from datetime import datetime
import numpy as np

def merge_dicts(dicts):
    """
    Merges a list of dicts by appending its values if keys are repeated.
    ====================================================================
    -Args:
        dicts [list] : list of dicts to be merged (numpy array not 
                supported yet).
    - Returns:
        merged_dict [dict] : resulting merged dictionary.
    ====================================================================
    """
    merged_dict = copy.deepcopy(dicts[0])
    if len(dicts) == 1:
        return merged_dict
    for dc in dicts[1:]:
        common_keys = [k for k in dc.keys() if k in merged_dict.keys()]
        for k, v in dc.items():
            if k in common_keys:
                if isinstance(merged_dict[k], list):
                    merged_dict[k].extend(v if isinstance(v, list) else [v])
                else:
                    merged_dict[k] = v if isinstance(v, list) else [v] + [merged_dict[k]]
            else:
                merged_dict.update({k : v})
    return merged_dict


def without_duplicates(iterable):
    """ Transforms an input Iterator into the same Iterator without duplicates.
    ===========================================================================
    - Args: 
        iterable [Iterator] : iterator to remove duplicates from.
    - Yields:
        Value of the input iterable if the value has not been already seen. 
    ============================================================================
    """
    visited = []
    for val in iterable:
        if val not in visited:
            visited.append(val)
            yield val

def flatten_dict(dic, merge_symbol=':'):
    """
    Flattens a tree like dictionary, merging root keys as the 
    concatenation of all branch keys with the specified merge_symbol (str).
    """
    flattened_dict = {}
    for key, val in dic.items():
        if isinstance(val, dict):
            inner_dict = flatten_dict(val)
            flattened_dict.update({key + merge_symbol + k_inner : val_inner \
                    for k_inner, val_inner in inner_dict.items()})
        else:
            flattened_dict.update({key : val})
    return flattened_dict

def key_of(dic, value):
    """ Returns the key corresponding to the specified value in the 
    supplied dict. """
    if isinstance(value, np.ndarray):
        return [k for k, v in dic.items() if all(v == value)][0]
    return [k for k, v in dic.items() if v == value][0]

class RegexpDict(dict):
    def __init(self, *args, **kwargs):
        super(RegexpDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        for regex_key, val in self.items():
            if re.match(regex_key, key):
                return val
        raise KeyError



def remove_duplicates(lst):
    """ Return input list without duplicates. """
    seen = []
    out_lst = []
    for elem in lst:
        if elem not in seen:
            out_lst.append(elem)
            seen.append(elem)
    return out_lst

def any_duplicates(iterable):
    """ Returns true if there are duplicate elements in list, dict keys, etc. """
    seen = []
    for elem in iterable:
        if elem in seen:
            return True
        seen.append(elem)
    return False

def isinstance_of_any(var, list_types):
    """ Checks if var is instance of any of the classes in the list. """
    return any([isinstance(var, type_elem) for type_elem in list_types])

def issubclass_of_any(var, list_types):
    """ Checks if var is instance of any of the classes in the list. """
    return any([issubclass(type(var), type_elem) for type_elem in list_types])

def save_pickle(dc, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(dc, f)

def load_pickle(filename):
    with open(filename + '.pickle', 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

class DataLogger:
    def __init__(self, fieldnames):
        self.data = {name : deque([]) for name in fieldnames}
        self.fieldnames = fieldnames

    def update(self, row_dict):
        for key in self.data.keys():
            self.data[key].append(row_dict[key])
            
    def save(self, experiment_name, num_robots):
        curr_date = datetime.now()
        file_name = 'spike_swarm_sim/logs/data/' + experiment_name\
                + str(num_robots) + '_robots_' + curr_date.strftime("%Y_%m_%d_%H_%M")
        with open(file_name + '.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in np.stack(*[self.data.values()]).T:
                writer.writerow({key : val for key, val in zip(self.fieldnames, row)})
