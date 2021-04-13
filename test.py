from sakura import RecNamespace


lossAccDict = {
    "loss": 0,
    "accuracy": 0
}

defaultMetrics = \
    {
        "test": lossAccDict,
        "train": lossAccDict,
    }


# d = RecNamespace({"current":None, "best":None})
#
# c = d
#
# c.current = 1
current = metrics.test.current
best = metrics.test.best
current.loss = 1
# best.loss = current.loss
vars(best).update(vars(current))

print(current)
print(best)
print(metrics.test.current)
print(metrics.test.best)