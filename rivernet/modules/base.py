import torch.nn as nn
from allennlp.common.registrable import Registrable


class Module(Registrable, nn.Module):
    pass
