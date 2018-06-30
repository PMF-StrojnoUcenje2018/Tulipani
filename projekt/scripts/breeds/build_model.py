import copy
import time
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model import *
from torchnet import meter
from torch.autograd import Variable

import sys
sys.path.append('scripts')
from breeds.data_loader import dset_classes, dset_loaders, dset_sizes, dsets

sys.path.append('utils')
from config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM, BATCH_SIZE
from logger import Logger

print('\nProcessing Model Breeds...\n')

classes_breeds = dsets['train'].classes


def to_np(x):
    return x.data.cpu().numpy()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


inputs, classes = next(iter(dset_loaders['train']))

out = make_grid(inputs)

# imshow(out, title=[dset_classes[x] for x in classes])


def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=LR_DECAY_EPOCH):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('Learning Rate: {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

best_acc = 0.0

def doEpoch(results, best_model, epoch, criterion, optimizer, lr_scheduler, num_epochs):
    global best_acc
    with open('results/breeds/model__breeds__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
        f.write(results)

    for phase in ['train', 'test']:
        if phase == 'train':
            optimizer = lr_scheduler(optimizer, epoch)
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        confusion_matrix = meter.ConfusionMeter(37)

        for data in dset_loaders[phase]:
            inputs, labels = data

            if torch.cuda.is_available():
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                score = model(inputs)
                confusion_matrix.add(score.data, labels.data)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.data[0]
            # running_corrects += torch.sum(preds == labels.data)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / dset_sizes[phase]
        epoch_acc = running_corrects / dset_sizes[phase]

        if phase == 'test':
            print("----------------------------- Confusion Matrix Classes -----------------------------")
            print(classes_breeds)
            print("----------------------------- Confusion Matrix Classes -----------------------------")
            print("")

        print('{} Loss: {:.8f} Acc: {:.8f}'.format(phase, epoch_loss, epoch_acc))

        results = ('{} Loss: {:.8f} Acc: {:.8f}\n'.format(phase, epoch_loss, epoch_acc)) + '\n'
        with open('results/breeds/model__breeds__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
            f.write(results)

        if phase == 'test':
            confusion = ('{}\n'.format(classes_breeds)) + '\n'
            with open('results/breeds/model__breeds__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
                f.write(confusion)
                cm_value = confusion_matrix.value()
                for i in cm_value:
                    for j in i:
                        f.write(str(j) + " ")
                    f.write("\n")
                    f.write("\n")
                f.write("\n")

        if phase == 'test' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model)

        if phase == 'test':
            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'loss': epoch_loss,
                'accuracy': epoch_acc
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch + 1)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), epoch + 1)
                logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)

            # (3) Log the images
            info = {
                'images': to_np(inputs.view(-1, 224, 224)[:25])
            }

            for tag, inputs in info.items():
                logger.image_summary(tag, inputs, epoch + 1)

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model = model

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)
        results = ('Epoch {}/{}\n'.format(epoch + 1, num_epochs)) + ('--' * 50) + '\n'
        doEpoch(results, best_model, epoch, criterion, optimizer, lr_scheduler, num_epochs)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:8f}\n'.format(best_acc))

    results = ('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)) + \
        ('Best val Acc: {:8f}\n'.format(best_acc))
    with open('results/breeds/model__breeds__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
        f.write(results)

    return best_model


def visualize_model(model, num_images=NUM_IMAGES):
    images_so_far = 0
    fig_num = 1

    plt.ioff()
    fig = plt.figure(fig_num)
    path = "./pics/breeds/testPics/"
    for i, data in enumerate(dset_loaders['test']):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
            imshow(inputs.cpu().data[j])

            if (i*BATCH_SIZE + j + 1) % num_images == 0 or j == inputs.size()[0] - 1:
                fig.savefig(path + str(fig_num) + "_fig.jpg")
                plt.close(fig)
                fig_num += 1
                fig = plt.figure(fig_num)
                images_so_far = 0


model = CNNModel()

logger = Logger('results/breeds/logs')

if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
visualize_model(model)

torch.save(model.cpu().state_dict(), 'results/breeds/model_breeds.pkl')