
from ..basic_params import get_param_parser


def get_params():
    parser = get_param_parser()
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--local_attention", type=bool, default=False)
    parser.add_argument("--resnet", type=bool, default=False)
    return parser.parse_args()
