from model import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import glob

import sys
sys.path.append('utils')
from config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM, BATCH_SIZE

sys.path.append('scripts')
from breeds.data_loader import dset_classes, dset_loaders, dset_sizes, dsets, transform

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.485, 0.456, 0.406])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

def visualize_new(listaSlika, model, num_images=NUM_IMAGES):

    images_so_far = 0
    fig_num = 1

    plt.ioff()
    fig = plt.figure(fig_num)
    path = "./pics/breeds/newPics/"

    lista = []
    for slika in listaSlika:
        lista.append(transform['test'](slika))

    tenzor = torch.stack(lista)
    inputs = Variable(tenzor.cuda())
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)

    for j in range(inputs.size()[0]):  
        images_so_far += 1
        ax = plt.subplot(num_images//2, 2, images_so_far)
        ax.axis('off')
        ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
        imshow(inputs.cpu().data[j])
    
        if ((j + 1) % num_images) == 0 or j == inputs.size()[0] - 1:
            fig.savefig(path + str(fig_num) + "_fig.jpg")
            plt.close(fig)
            fig_num += 1
            fig = plt.figure(fig_num)
            images_so_far = 0

def to_np(x):
    return x.data.cpu().numpy()

if __name__ == '__main__':
    model = CNNModel()
    model.cuda().load_state_dict(torch.load('results/breeds/model_breeds.pkl'))
    model.eval()

    print(model)

    imageList = []
    for filename in glob.glob('testneSlike/*.jpg'):
        im = Image.open(filename)
        imageList.append(im)

    visualize_new(imageList, model)
    
