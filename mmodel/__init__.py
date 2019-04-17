
def get_module(name):
    name = name.upper()
    
    if name == 'BY':
        from .Bayes import model
        return model.param, model.BayesModel()
    elif name == 'DROP':
        from .dropout import model
        return model.param, model.Dropout()
    


def get_params():
    from .basic_params import get_param_parser
    return get_param_parser().parse_args()