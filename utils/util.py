# Created by Xingyu Lin, 2018/11/19
import numpy as np


def get_name_arr_and_len(field_indexer, dim_idx):
    """Returns a string array of element names and the max name length."""
    axis = field_indexer._axes[dim_idx]
    size = field_indexer._field.shape[dim_idx]
    try:
        name_len = max(len(name) for name in axis.names)
        name_arr = np.zeros(size, dtype=object)
        for name in axis.names:
            if name:
                # Use the `Axis` object to convert the name into a numpy index, then
                # use this index to write into name_arr.
                name_arr[axis.convert_key_item(name)] = name
    except AttributeError:
        name_arr = np.zeros(size, dtype=object)  # An array of zero-length strings
        name_len = 0
    return name_arr, name_len
