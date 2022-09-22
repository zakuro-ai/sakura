from gnutools.utils import RecNamespace

lossAcc = {"loss": 0, "accuracy": 0}
metrics = {"current": lossAcc.copy(), "best": lossAcc.copy()}
defaultMetrics = RecNamespace(
    {"train": metrics.copy(), "test": metrics.copy()})

lossWerCer = {"loss": 0, "wer": 100, "cer": 100}
metrics = {"current": lossWerCer.copy(), "best": lossWerCer.copy()}
asr_metrics = RecNamespace({"train": metrics.copy(), "test": metrics.copy()})
