import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy


def load_dataset():
    # data_path = 'pytorch_dataset/train/'
    data_path = 'pytorch_dataset/train'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True
    )

    data_path = 'pytorch_dataset/val'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        num_workers=0,
        shuffle=True
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
    from net import vgg11_bn

    model = vgg11_bn()
    train_loader, test_loader = load_dataset()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    trained_model, val_acc_history, acc, loss = train_model(model, train_loader, test_loader, optimizer, 25)
    torch.save(trained_model.state_dict(), 'loss_{:4f}__acc_{:4f}.pth.tar'.format(loss, acc))
