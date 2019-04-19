from ..basic_params import parser

def get_params():
    parser.add_argument("--class_num", type=int, default=10)
    parser.add_argument("--sigma1", type=float, default=-0)
    parser.add_argument("--sigma2", type=float, default=-6)
    parser.add_argument("--pi", type=float, default=0.5)
    parser.add_argument("--samples", type=float, default=10)
    parser.add_argument("--trainsamples", type=float, default=2)
    return parser.parse_args()
