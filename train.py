import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
import copy
import cv2
from icon_dataset import IconDataset


def load_dataset(data_dir, in_size):
    train_dataset = IconDataset(root=os.path.join(data_dir), length=2000,
                                transform=transforms.Compose([
                                    # transforms.RandomCrop((im_h - 3, im_w - 3), padding=3),
                                    transforms.Resize((in_size, in_size)),
                                    transforms.ToTensor(), ]),
                                )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True
    )

    test_dataset = IconDataset(root=os.path.join(data_dir), length=200,
                               transform=transforms.Compose([
                                   # transforms.RandomCrop(in_size - 6, padding=3),
                                   transforms.Resize((in_size, in_size)),
                                   transforms.ToTensor(), ])
                               )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=8,
        shuffle=True,
    )

    return train_loader, test_loader


def train_model(model, train_data_loader, test_data_loader, optimizer, num_epochs=25, is_inception=False):
    model = model.to('cuda')
    since = time.time()

    val_acc_history = []
    lossFunc = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_acc_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch_idx, (inputs, target) in enumerate(train_data_loader):
                inputs = inputs.to('cuda')
                target = target.to('cuda')

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs = model(inputs)
                        loss = lossFunc(outputs, target)
                    else:
                        outputs = model(inputs)
                        loss = lossFunc(outputs, target)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == target.data)

            epoch_loss = running_loss / len(test_data_loader.dataset)
            epoch_acc = running_corrects.double() / len(test_data_loader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc_loss, best_acc


if __name__ == '__main__':
    import os
    from net import VGG

    the_lr = 0.001

    # #######################################################################################################
    # inSize = 64
    # icon = 'gun_scope'
    #
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(icon, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, inSize)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 20)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
    #
    # ########################################################################################################
    # in_size = 64
    # icon = 'gun_muzzle'
    #
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(in_size, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, in_size)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 20)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
    #
    # ########################################################################################################
    # in_size = 64
    # icon = 'gun_grip'
    #
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(in_size, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, in_size)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 5)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
    # # #########################################################################################################
    in_size = 64
    icon = 'gun_butt'

    dataRoot = os.path.join("dataSets", icon, "train")
    out_num = len(os.listdir(dataRoot))

    model = VGG(in_size, out_num)
    train_loader, test_loader = load_dataset(dataRoot, in_size)
    optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)

    model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 10)
    print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
    #
    # # ########################################################################################################
    # in_size = 64
    # icon = 'gun_name'
    #
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(in_size, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, in_size)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 20)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
    #
    # #########################################################################################################
    # in_size = 32
    # icon = 'fire_mode'
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(icon, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, in_size)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 10)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))

    #########################################################################################################
    # in_size = 32
    # icon = 'in_tab'
    # dataRoot = os.path.join("dataSets", icon, "train")
    # out_num = len(os.listdir(dataRoot))
    #
    # model = VGG(in_size, out_num)
    # train_loader, test_loader = load_dataset(dataRoot, in_size)
    # optimizer = optim.SGD(model.parameters(), lr=the_lr, momentum=0.9)
    #
    # model.load_state_dict(torch.load('{}.pth.tar'.format(icon)))
    # trained_model, val_acc_history, loss, acc = train_model(model, train_loader, test_loader, optimizer, 20)
    # print('{}_loss_{:2f}__acc_{:2f}.pth.tar'.format(icon, loss, acc))
    # torch.save(trained_model.state_dict(), '{}.pth.tar'.format(icon))
