from argparse import Namespace

def RecNamespace(d):
    results = {}
    for k, v in d.items():
        if type(v) == dict:
            v = RecNamespace(v)
        results[k] = v
    return Namespace(**results)


def RecDict(d):
    results = {}
    for k, v in vars(d).items():
        if type(v) == Namespace:
            v = RecDict(v)
        results[k] = v
    return dict(results)