from data.bird_helper import CUBHelper
from data.car_helper import CarHelper
from data.Air_helper import AirHelper
from data.animal_helper import AWA2Helper
from data.webvision_helper import WebVisionHelper
from data.apy_helper import APYHelper
from data.sun_helper import SUNHelper
from data.flo_helper import FLOHelper


def get_data_helper(args):
    if 'CUB' in args.data_path:
        return CUBHelper(args)
    elif 'Car' in args.data_path:
        return CarHelper(args)
    elif 'Air' in args.data_path:
        return AirHelper(args)
    elif 'WebVision' in args.data_path:
        return WebVisionHelper(args)
    elif 'AWA2' in args.data_path:
        return AWA2Helper(args)
    elif 'APY' in args.data_path:
        return APYHelper(args)
    elif 'SUN' in args.data_path:
        return SUNHelper(args)
    elif 'FLO' in args.data_path:
        return FLOHelper(args)
    else:
        raise NotImplementedError
