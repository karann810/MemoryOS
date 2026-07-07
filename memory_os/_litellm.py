# Internal shim for litellm when upstream package isn't available (for tests)
suppress_debug_info = False

class _Resp:
    def __init__(self, data):
        self.data = data


def embedding(model: str, input: list[str]):
    vec = [0.0] * 1536
    return _Resp(data=[{"embedding": vec}])


class _ChatCompletions:
    def create(self, *args, **kwargs):
        class R:
            facts = []
        return R()


class _Completion:
    def __init__(self):
        self.chat = self
        self.completions = _ChatCompletions()

    def __getattr__(self, name):
        if name == 'completions':
            return self.completions
        raise AttributeError(name)

completion = _Completion()
