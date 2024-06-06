class ProgressPrinter(object):
    def __init__(self, *header, silent=False):
        super().__init__()
        self.rawheader = header
        self.width = max(14, max(len(h) + 8 for h in self.rawheader))
        self.autoprint = True
        self.extra = None
        self.silent = silent
        self.offset = 0

    def addobs(self, *observation):
        for n, v in enumerate(observation):
            if v is not None:
                self.n[n] += 1
                self.sum[n] += v
                self.nsincelast[n] += 1
                self.sincelast[n] += v

        self.cnt += 1
        if self.autoprint and self.cnt and (self.cnt & (self.cnt - 1)) == 0:
            self.print()

    def format_time(self, dt):
        if dt < 1:
            return f'{1000*dt:>4.3g} ms'
        elif dt < 60:
            return f'{dt:>5.3g} s'
        elif dt < 60 * 60:
            return f'{dt/60:>5.3g} m'
        elif dt < 24 * 60 * 60:
            return f'{dt/(60*60):>5.3g} h'
        elif dt < 7 * 24 * 60 * 60:
            return f'{dt/(24*60*60):>5.3g} d'
        else:
            return f'{dt/(7*24*60*60):>5.3g} w'

    def peek_since_last(self, n):
        return self.sincelast[n] / self.nsincelast[n] if self.nsincelast[n] else None

    def print(self):
        if not self.silent and any(self.nsincelast):
            import time

            end = time.time()

            print(' '.join([ f'{self.cnt+self.offset:<7d}' ] +
                           [ f'{v[0]:{self.width-8}.3f} ({v[1]:4.3f})'
                             for n, s in enumerate(self.rawheader)
                             for v in ((self.sum[n]/max(1,self.n[n]), self.sincelast[n]/max(1,self.nsincelast[n]),),)
                           ] +
                           [ self.format_time(end - self.start) ]),
                  flush=True)
            self.nsincelast = [0] * len(self.rawheader)
            self.sincelast = [0] * len(self.sincelast)

            if callable(self.extra):
                self.extra()

    def __enter__(self):
        import time

        self.fullheader = ['n'] + [ f'{what} (since)' for what in self.rawheader ] + ['dt']
        if not self.silent:
            print(' '.join([ f'{h:<7s}' if n == 0 else f'{h:>7s}' if h == 'dt' else f'{h:>{self.width}s}' for n, h in enumerate(self.fullheader) ]), flush=True)
        self.cnt = 0
        self.n = [0] * len(self.rawheader)
        self.sum = [0] * len(self.rawheader)
        self.nsincelast = [0] * len(self.rawheader)
        self.sincelast = [0] * len(self.rawheader)
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.print()
