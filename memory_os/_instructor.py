# Minimal shim for `instructor` used in tests
# Provides a `from_litellm` helper that simply wraps the provided completion

class _ClientWrapper:
    def __init__(self, completion):
        self.chat = completion


def from_litellm(completion):
    # Return a simple wrapper that exposes `.chat.completions.create(...)`
    return _ClientWrapper(completion)
