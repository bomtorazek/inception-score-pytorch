import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import argparse
import os

from torchvision.models.inception import inception_v3

import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
from dataset import GeneralDataset


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-classes', type = int, default=1000)
    parser.add_argument('--model-path', default='')
    parser.add_argument('--gpu', type = str, default='0')
    args = parser.parse_args()
    return args



def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)#type(dtype)
    if args.num_classes != 1000:
        inception_model.fc = nn.Linear(inception_model.fc.in_features, args.num_classes)
   
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location='cuda:0')['state_dict']
        device = torch.device("cuda:0")
        keys= list(state_dict.keys())
        for old_key in keys:
            new_key = old_key[9:]
            state_dict[new_key] = state_dict.pop(old_key)
        inception_model = inception_model.to(device)
        inception_model.load_state_dict(state_dict)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').to(device)#.type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, args.num_classes))
    
    for i, batch in enumerate(tqdm(dataloader, total=int(len(dataloader)))):
        batch = batch.to(device) # type(dtype)
        # batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
            #If qk is not None, then compute the Kullback-Leibler divergence
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    
    args = argument()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    dataset_4IS = GeneralDataset(args.path,transform=transforms.Compose([
                                 transforms.Resize(299),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                             ]))


    print("Calculating Inception Score...")
    print(inception_score(dataset_4IS, cuda=True, batch_size=args.batch_size, resize=True, splits=10))
