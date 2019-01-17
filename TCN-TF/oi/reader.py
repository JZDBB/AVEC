from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Reader(object):
    
    def __init__(self, batch_size, fake, **kwargs):
        self.batch_size = batch_size
        self.fake = fake
    
    @abstractmethod
    def read(self):
        net_inputs = {}
        ground_truth = {}
        return net_inputs, ground_truth
