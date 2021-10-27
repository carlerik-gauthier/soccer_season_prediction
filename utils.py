import os

MODEL_TYPE = ('regression', 'classification', 'ranking')

def is_available(module: str, name: str):
    """
    :param module: str: name of the module
    :param name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    return False

def retrieve_model(module: str, name: str):
    """
    :param module: str: name of the module
    :param name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    if not is_available(module=module, name=name):
        raise ValueError("{name} is not available in {module}".format(name=name, module=module)
        )
    
    pass

def train_model(model_type: str):
    """
    :param model_type: str: type of rank predicter. Must be either 'regression', 'classification' or 'ranking'

    :returns: scikit-learn model
    """
    if model_type not in MODEL_TYPE:
        raise ValueError("model_type MUST be one the following values : {values}".format(values=', '.join(MODEL_TYPE))
        )
    
    # save model
    model_name = "{model_type}_ranker".format(model_type=model_type)
    pass
