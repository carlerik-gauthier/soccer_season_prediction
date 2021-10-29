import os
import pickle

MODEL_TYPE = ('regression', 'classification', 'ranking')

def is_available(module_path: str, file_name: str):
    """
    :param module_path: str: path to the module
    :param file_name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    # mypath = os.path.join(os.getcwd(), 'misc')
    onlyfiles = [f for f in os.listdir(module_path) if os.path.isfile(os.path.join(module_path, f))]
    return False

def retrieve_model(module_path: str, file_name: str):
    """
    :param module: str: path to the the module
    :param file_name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    if not is_available(module_path=module_path, name=file_name):
        raise ValueError("{name} is not available in {module}".format(name=file_name, module=module_path)
        )
    # load model
    model = pickle.load(open(file_name, 'rb'))
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
    if model_type == 'regression':
        # from rank_predictor.regression import
        model = ...
        pass
    elif model_type == 'classification':
        from rank_predictor.classification import get_gradient_boosting_classifier_ranker
        model = ...
        pass
    else:
        from rank_predictor.ranking import get_xgboost_rank_ranker 
        model = ...

    
    # save model
    pickle.dump(model, open(model_name+'.pickle', 'wb'))

    return model