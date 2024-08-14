
from multiprocessing import Process, Queue

class ProcessBaseClass(Process):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        