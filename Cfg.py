
class Cfg(object):

    def __init__(self, dict):
        self.__dict__.update(dict)

    def __str__(self):
        return "\n".join(f"{k}\t{v}" for k, v in self.__dict__.items())