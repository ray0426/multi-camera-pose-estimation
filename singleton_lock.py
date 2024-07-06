import threading

class SingletonLock:
    _locks = {}

    @classmethod
    def get_lock(cls, name):
        if name not in cls._locks:
            cls._locks[name] = threading.Lock()
        return cls._locks[name]