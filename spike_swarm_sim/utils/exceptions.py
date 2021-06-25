import logging
class ConfigException(Exception):
    def __init__(self, msg):
        self.msg = 'Configuration Exception: ' + msg
        super().__init__(logging.error(self.msg))

class JSONParserException(Exception):
    def __init__(self, msg):
        self.msg = 'JSON Parser Exception: ' + msg
        super().__init__(logging.error(self.msg))

class ShapeMismatchException(Exception):
    def __init__(self, msg):
        self.msg = 'Shape Mismatch Exception: ' + msg
        super().__init__(logging.error(self.msg))