import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
#from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2

import datasets.mvtec as mvtec

from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
#from pyod.models.auto_encoder import auto_encoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.sos import SOS
#from pyod.models.xgbod import XGBOD

from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


args = parse_args()

# device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Load model ...')
# load model
model = wide_resnet50_2(pretrained=True, progress=True)
model.to(device)
model.eval()

print('Set models intermediate outputs ...')
# set model's intermediate outputs
outputs = []
def hook(module, input, output):
    outputs.append(output)
model.layer1[-1].register_forward_hook(hook)
model.layer2[-1].register_forward_hook(hook)
model.layer3[-1].register_forward_hook(hook)
model.avgpool.register_forward_hook(hook)

os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

print(mvtec.CLASS_NAMES, '\n')
for n, class_name in enumerate(mvtec.CLASS_NAMES):
    if n > 0: break
    print('\n######################',n , class_name, '######################\n')
    dataset = mvtec.MVTecDataset(class_name=class_name)
    dataloader = DataLoader(dataset, batch_size=32, pin_memory=True, shuffle=False)
    
    model_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

    print('\nextract train set features ...\n')
    # extract train set features
    feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)
    if not os.path.exists(feature_filepath):
        for (x, ids) in tqdm(dataloader, '| feature extraction | train | %s |' % class_name):
            
            #print(ids)
            
            # model prediction
            with torch.no_grad():
                pred = model(x.to(device)) # No need for preds, needed to initalize hooks
            
            # get intermediate layer outputs
            for k, v in zip(model_outputs.keys(), outputs):
                model_outputs[k].append(v)
            
            # initialize hook outputs
            outputs = []
        
        for k, v in model_outputs.items():
            model_outputs[k] = torch.cat(v, 0)
        
        # save extracted feature
        with open(feature_filepath, 'wb') as f:
            pickle.dump(model_outputs, f)
    else:
        print('load train set feature from: %s' % feature_filepath)
        with open(feature_filepath, 'rb') as f:
            model_outputs = pickle.load(f)

    print('\n\n----------------------------------------')
    print('Train Outputs:')
    print(model_outputs['layer1'].size())
    print(model_outputs['layer2'].size())
    print(model_outputs['layer3'].size())
    print(model_outputs['avgpool'].size())
    print('----------------------------------------\n\n')

    labels = dataset.get_labels()
    file_list = dataset.load_dataset_folder()

    od_models = [COPOD(), KNN(), ABOD(), CBLOF(), COF(), HBOS(), IForest(), LMDD(),
                 LODA(), LOF(), LOCI(), MCD(), OCSVM(), PCA(), SOD(), SOS()]
    
    od_names = ['COPOD', 'KNN', 'ABOD', 'CBLOF', 'COF', 'HBOS', 'IForest', 'LMDD',
                 'LODA', 'LOF', 'LOCI', 'MCD', 'OCSVM', 'PCA', 'SOD', 'SOS']
    

    for n, (model, name) in enumerate(zip(od_models, od_names)):
        clf = model
        
        encoding = model_outputs['avgpool'].detach().cpu().numpy().squeeze()
        
        clf.fit(encoding)
        pred = clf.predict(encoding)
        scores = clf.decision_function(encoding)
        proba = clf.predict_proba(encoding)
        evaluate_print(name, labels, scores)
    
        fpr, tpr, thresholds = roc_curve(labels, proba[:, 1], pos_label = 1)
    
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

