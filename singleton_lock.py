import threading

class SingletonLock:
    _locks = {}

    @classmethod
    def get_lock(cls, name):
        if name not in cls._locks:
            cls._locks[name] = threading.Lock()
        return cls._locks[name]

print_lock = SingletonLock.get_lock('print')

def tprint(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)