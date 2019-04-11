
from ..basic_params import parser
from mtrain.watcher import watcher

def get_params():
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--local_attention", type=bool, default=False)
    parser.add_argument("--to", type=int, default=10)
    
    args = parser.parse_args()
    watcher.parameter_note(args)
    return args
