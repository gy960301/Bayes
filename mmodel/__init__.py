
def get_module(name):
    name = name.upper()
    
    if name == 'BY':
        from .Bayes import model
        return model.param, model.BayesModel()
    elif name == 'DROP':
        from .dropout import model
        return model.param, model.Dropout()
    elif name == 'BY800':
        from .Bayes import model800
        return model800.param, model800.BayesModel()
    elif name == 'BY1200':
        from .Bayes import model1200
        return model1200.param, model1200.BayesModel()
    


def get_params():
    from .basic_params import get_param_parser
    return get_param_parser().parse_args()