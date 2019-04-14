cache = {}


def update_cache(sentence, response):
    global cache
    resp = cache.get(sentence)
    if (resp is None):
        cache[sentence] = response
    pass


def get_cahce(sentence):
    resp = cache.get(sentence)

    if (resp is not None):
        return resp
    else:
        return None


def clear_cache():
    cache.clear()
    pass
