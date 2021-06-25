class Globals:
    def __init__(self):
        self._EVAL = False
        self._DEBUG = False
        self._RENDER = True
        self._INFO = True

    def set_eval_state(self, new_state):
        self._EVAL = new_state
    
    def set_render_state(self, new_state):
        self._RENDER = new_state

    def set_debug_state(self, new_state):
        self._DEBUG = new_state

    def set_states(self, render=True, eval=False, debug=False, info=False):
        self._EVAL = eval
        self._RENDER = render
        self._DEBUG = debug
        self._INFO = info

    @property
    def EVAL(self):
        return self._EVAL
    
    @property
    def RENDER(self):
        return self._RENDER

    @property
    def DEBUG(self):
        return self._DEBUG

    @property
    def INFO(self):
        return self._INFO or self._DEBUG

global_states = Globals()