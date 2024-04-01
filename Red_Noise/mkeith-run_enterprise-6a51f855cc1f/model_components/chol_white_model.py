from . import white_model
from .white_model import name,argdec,setup_argparse



def setup_model(args, psr, parfile):
    wmodel = white_model.setup_model(args,psr,parfile)(psr)
    return wmodel


