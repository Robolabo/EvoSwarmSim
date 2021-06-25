from functools import wraps

""" language_dict gathers all the syntax and possible queries of the 
phenotype-genotype interface of the simulator.
"""
language_dict = {}

def GET(query):
    """ Decorator that marks a method as a GET query handler for some 
    variable in the genotype-phenotype interface. The decorator 
    receives as argument the name of the query. For instace, this 
    decorator is applied to the Synapses.get_weights method as 
    @GET("synapses:weights").
    """
    # @wraps()
    def _GET(func):
        query_elements = ['GET'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _GET 

def SET(query):
    """ Decorator that marks a method as a SET query handler for some 
    variable in the genotype-phenotype interface. The decorator 
    receives as argument the name of the query. For instace, this 
    decorator is applied to the Synapses.set_weights method as 
    @SET("synapses:weights").
    """
    def _SET(func):
        query_elements = ['SET'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)        
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _SET

def LEN(query):
    """ Decorator that marks a method as a LET query handler for some 
    variable in the genotype-phenotype interface. The decorator 
    receives as argument the name of the query. For instace, this 
    decorator is applied to the Synapses.len_weights method as 
    @LEN("synapses:weights").
    """
    def _LEN(func):
        query_elements = ['LEN'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _LEN

def INIT(query):
    """ Decorator that marks a method as an INIT query handler for some 
    variable in the genotype-phenotype interface. The decorator 
    receives as argument the name of the query. For instace, this 
    decorator is applied to the Synapses.init_weights method as 
    @INIT("synapses:weights").
    """
    def _INIT(func):
        query_elements = ['INIT'] + query.split(':')
        aux_dict = language_dict
        for i, elem in enumerate(query_elements):
            if elem not in aux_dict.keys():
                aux_dict[elem] = {}
            if i < len(query_elements) - 1:
                aux_dict = aux_dict[elem]
            else:
                aux_dict[elem] = func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return _INIT