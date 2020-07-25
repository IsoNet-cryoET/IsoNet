class Arg:
    def __init__(self,dictionary):
        for k, v in dictionary.items():
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            setattr(self, k, v)
            