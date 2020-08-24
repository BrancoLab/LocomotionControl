from .first_order_lag import FirstOrderLagModel
from .two_wheeled import TwoWheeledModel
from .cartpole import CartPoleModel

def make_model(args, config):
    
    if args.env == "FirstOrderLag":
        return FirstOrderLagModel(config)
    elif args.env == "TwoWheeledConst" or args.env == "TwoWheeledTrack":
        return TwoWheeledModel(config)
    elif args.env == "CartPole":
        return CartPoleModel(config)
    
    raise NotImplementedError("There is not {} Model".format(args.env))