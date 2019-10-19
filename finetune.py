import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import os
import random
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_transform = transforms.Compose([
        # transforms.CenterCrop(299),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def init_random_seed(manual_seed):
    """Init random seed."""
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def figure(epoch, Acc_train, Acc_val, Loss_train, Loss_val):
    x1 = range(0, epoch)
    x2 = range(0, epoch)
    x3 = range(0, epoch)
    x4 = range(0, epoch)
    y1 = Acc_train
    y2 = Acc_val
    y3 = Loss_train
    y4 = Loss_val
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-', color='red', label='Train acc')
    plt.plot(x2, y2, 'o-', color='blue', label='Val acc')
    plt.title('accuracy vs. epoches')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right', shadow=True, fontsize='x-small')
    plt.subplot(2, 1, 2)
    plt.plot(x3, y3, '.-', color='red', label='Train loss')
    plt.plot(x4, y4, '.-', color='blue', label='Val loss')
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.savefig("./ckpt/Figure_1.png")
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    ckpt_index = 0
    Loss_train = []
    Loss_val = []
    Acc_train = []
    Acc_val = []
    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}'.format(epoch))
        result.writelines('\nEpoch {}'.format(epoch))
        print('-' * 10)
        result.writelines('\n' + '-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            flag = True
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                flag = False
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if use_gpu:
                    inputs = Variable(inputs.cuda(), volatile=flag)
                    labels = Variable(labels.cuda(), volatile=flag)
                else:
                    inputs, labels = Variable(inputs, volatile=flag), Variable(labels, volatile=flag)

                # zero the parameter gradients
                optimizer.zero_grad()

                # outputs = model(inputs)

                if phase == 'train':
                    outputs, aux_outputs = model(inputs)
                else:
                    outputs = model(inputs)

                # print(outputs)
                _, preds = torch.max(outputs, 1)
                # print(preds)

                if phase == 'train':
                    # loss1 = criterion(outputs, labels)
                    # loss2 = criterion(aux_outputs, labels)
                    # loss = loss1 + 0.4*loss2
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.data[0]
                running_corrects += (torch.sum(preds == labels).data[0])

            epoch_loss = running_loss*1.0/dataset_sizes[phase]
            epoch_acc = running_corrects*1.0/dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            result.writelines('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                Loss_train.append(epoch_loss)
                Acc_train.append(100 * epoch_acc)
            else:
                Loss_val.append(epoch_loss)
                Acc_val.append(100 * epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), './ckpt/finetune_' + str(ckpt_index) + '.pth')
                ckpt_index += 1
                best_model_wts = model.state_dict()
                print('Best val Acc: {:4f}'.format(best_acc))
                result.writelines('\nBest val Acc: {:4f}'.format(best_acc))
        time_elapsed = time.time() - since
        print('epoch ' + str(epoch) + ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result.writelines('\nepoch ' + str(epoch) + ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))
    result.writelines('\nBest val Acc: {:4f}'.format(best_acc))
    result.flush()

    figure(num_epochs, Acc_train, Acc_val, Loss_train, Loss_val)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    manual_seed = None
    seed = init_random_seed(manual_seed)
    use_gpu = torch.cuda.is_available()

    data_dir = '/home/jiaojiao/patch/One-Shot/cam'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform)
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    global result
    result = open('./ckpt/Figure_1.txt', 'a')
    result.writelines("use random seed: " + str(seed))
    lr = 0.0005  # can't too bigger
    print('lr = ' + str(lr))
    result.writelines('\nlr = ' + str(lr))
    weight_decay = 0.005
    print('weight_decay = ' + str(weight_decay))
    result.writelines('\nweight_decay = ' + str(weight_decay))
    result.flush()
    momentum = 0.5
    print('momentum = ' + str(momentum))
    result.writelines('\nmomentum = ' + str(momentum))

    model = torchvision.models.inception_v3(pretrained=False, num_classes=1)
    # model = torchvision.models.vgg19_bn(pretrained=False, num_classes=2)
    # model = torchvision.models.resnet50(pretrained=False, num_classes=2)

    # model.load_state_dict(torch.load('./rank/Figure_3.pth'))
    model.load_state_dict(torch.load('./RANK/Figure_2.pth'))
    fc_num = model.fc.in_features
    model.fc = nn.Linear(fc_num, 2)

    # Auxlogits_params = list(map(id, model.AuxLogits.parameters()))
    # Mixed_7a = list(map(id, model.Mixed_7a.parameters()))
    # Mixed_7b = list(map(id, model.Mixed_7b.parameters()))
    # Mixed_7c = list(map(id, model.Mixed_7c.parameters()))
    fc_params = list(map(id, model.fc.parameters()))
    # fc_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in fc_params, model.parameters())

    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    param = [
        {"params": base_params},
        # {"params": model.AuxLogits.parameters(), 'lr': lr*5},
        # {"params": model.Mixed_7a.parameters(), 'lr': lr * 10},
        # {"params": model.Mixed_7b.parameters(), 'lr': lr * 10},
        # {"params": model.Mixed_7c.parameters(), 'lr': lr * 10},
        # {"params": model.fc.parameters(), 'lr': lr * 10},
        {"params": model.fc.parameters(), 'lr': lr * 10},
    ]

    # Set SGD + Momentum
    optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
    # Decay LR by a factor of 0.5 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=15)
    result.close()



