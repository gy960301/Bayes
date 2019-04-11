import inspect

def get_args_dict(function):

    args_spec = inspect.getfullargspec(function)
    args_name = args_spec.args
    args_default = args_spec.defaults

    args_default =  (None,) * (len(args_name) - len(args_default)) + args_default

    d = dict(zip(args_name, args_default))
    d.pop('self')
    return d