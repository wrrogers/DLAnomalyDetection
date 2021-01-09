import os
import sys

class Hardware:
    def __init__(self):
        self.device = 'cuda'
        self.sample_device = 'cpu'
        self.available_gpu = '6'
        self.n_gpu = 1
        port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        )
        self.dist_url = f"tcp://127.0.0.1:{port}"


class Model:
    def __init__(self):
        self.channel = 1
        
        
class Data:
    def __init__(self):
        self.size = 512
        #self.set = 'CelebA'
        self.set = 'CIFAR10'
        #self.set = 'MNIST'
        self.path = r'F:\William\dataset_HDF5'
        self.sample_size=2


class Optim:
    def __init__(self):
        self.lr = 0.0003
        self.sched = 'cycle'
        
        
class Train:
    def __init__(self):
        self.batch_size = 32
        self.n_epochs = 16
        self.log_iter = 1
        self.checkpoint = 'checkpoint/stop_2.pt'


class Config:
    def __init__(self):
        self.hardware = Hardware()
        self.model = Model()
        self.data = Data()
        self.optim = Optim()
        self.train = Train()

