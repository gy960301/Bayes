
from ..basic_params import parser


def get_params():
    parser = get_param_parser()
    #number of burn-in round,start averaging after num_burn round
    parser.add_argument("--num_burn", type=int, default=1000)
    #initial gaussian standard deviation used in weight init
    parser.add_argument("--init_sigma", type=float, default=0.001)
    #random number seed
    parser.add_argument("--seed", type=int, default=0)
    #start sampling weight after this round
    parser.add_argument("--start_sample", type=int, default=1)
    parser.add_argument("--start_hsample", type=int, default=1)
    #Gamma(alpha,beta) prior on regularizer
    parser.add_argument("--hyper_alpha", type=float, default=1.0)
    parser.add_argument("--hyper_beta", type=float, default=1.0)
    #sample hyper parameter each gap_hsample over training data
    parser.add_argument("--gap_hsample", type=int, default=1)
    parser.add_argument("--num_class", type=int, default=10)
     #following things are not set by users
        #sample weight
        self.wsample=1.0
        #round counter
        self.rcounter=0




    return parser.parse_args()
