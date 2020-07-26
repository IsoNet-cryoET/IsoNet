class Arg:
    def __init__(self,dictionary):
        param_list = []
        for k, v in dictionary.items():
            if k not in param_list:
                pass
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            setattr(self, k, v)
            