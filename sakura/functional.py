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

lossAcc = {"loss": 0, "accuracy": 0}
metrics = {"current": lossAcc.copy(), "best": lossAcc.copy()}
defaultMetrics = RecNamespace({"train": metrics.copy(), "test": metrics.copy()})

lossWerCer = {"loss": 0, "wer": 100, "cer": 100}
metrics = {"current": lossWerCer.copy(), "best": lossWerCer.copy()}
asr_metrics = RecNamespace({"train": metrics.copy(), "test": metrics.copy()})
