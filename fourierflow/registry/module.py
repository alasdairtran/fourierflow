import torch.nn as nn
from allennlp.common import Params, Registrable


class Module(Registrable, nn.Module):
    ...
