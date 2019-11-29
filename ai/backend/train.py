import copy
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import BATCH_SIZE
from config import DATASET_PATH
from config import EPOCHS
from utils import get_data_loader
from utils import get_device
from utils import get_metadata
from utils import get_net


def main(net, device, checkpoint, dataloaders, writer=None, epochs=10, lr=1e-3):
    """Transfer Learning for Computer Vision.

    Args:
        net: model instance.
        device (torch.device): where data will be put on.
        checkpoint: path to checkpoint.
        dataloaders (dict): dict mapping keys to corresponding dataloaders
        writer (SummaryWriter, optional): tensorboard writer.
        epochs: number of epochs to train the model.
        lr: learning rate.

    Returns:
        net: model instance.
    """
    since = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    initial_epoch = 0
    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1

    best_acc = 0.0
    for epoch in range(initial_epoch, epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i, data in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels)

                if writer is not None:
                    writer.add_scalar(f'{phase}_loss', loss, epoch * len(dataloaders[phase]) + i)
                #     writer.add_figure('predictions vs. actual',
                #                       plot_classes_preds(net, class_names, inputs, labels),
                #                       global_step=epoch * len(dataset_loader) + i)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase} Epoch: {epoch} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}\n')

            # Saving checkpoint
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_net_wts = copy.deepcopy(net.state_dict())
                torch.save({
                    'epoch': epoch,
                    'acc': epoch_acc,
                    'loss': epoch_loss,
                    'model_state_dict': best_net_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint)

            time.sleep(0.25)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best net weights
    net.load_state_dict(best_net_wts)
    return net


if __name__ == '__main__':

    checkpoint_path = os.path.join('checkpoint', 'checkpoint.pth')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # TensorBoard setup
    log_path = os.path.join('runs', 'experiment_1')
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    data_loaders = {x: get_data_loader(os.path.join(DATASET_PATH, x), batch_size=BATCH_SIZE, mode=x)
                    for x in ['train', 'val']}

    # torch.multiprocessing.freeze_support()
    device = get_device()
    _, num_classes, _ = get_metadata(os.path.join(DATASET_PATH, 'train'))
    model = get_net(model_name='mobilenet_v2', mode='train', device=device,
                    pretrained=True, num_classes=num_classes)
    model = main(model, device, checkpoint_path, data_loaders, writer=writer, epochs=EPOCHS)
    writer.close()
