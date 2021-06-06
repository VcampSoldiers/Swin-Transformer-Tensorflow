import collections

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)