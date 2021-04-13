class range:
    def __init__(self,
                 start=1,
                 stop=100):
        self.start = start
        self.stop = stop
        self.current = self.start
        self.best = self.start
        self.current = self.start
        self.total = self.stop - self.start

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current <self.stop:
            return self.current
        else:
            raise StopIteration
