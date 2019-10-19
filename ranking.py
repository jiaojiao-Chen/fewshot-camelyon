import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import time
import random
from PIL import ImageFile
from dataloader.diferNormalize import TargetDataset
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True


big_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

small_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=299, scale=(0.4, 0.9)),
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
    plt.legend(loc='upper left', shadow=True, fontsize='x-small')
    plt.subplot(2, 1, 2)
    plt.plot(x3, y3, '.-', color='red', label='Train loss')
    plt.plot(x4, y4, '.-', color='blue', label='Val loss')
    plt.xlabel('loss vs. epoches')
    plt.ylabel('loss')
    plt.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.savefig("../ckpt/Figure_1.png")
    plt.show()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            # for big_data, small_data in data_loader[phase]:
            for i, (big_data, small_data) in enumerate(data_loader):
                i += 1
                if i == 2000:    # 1000
                    break
                labels = torch.ones(big_data.size(0)).unsqueeze(0).float()

                if use_gpu:
                    big_data = Variable(big_data.cuda())
                    small_data = Variable(small_data.cuda())

                    labels = Variable(labels.cuda()).view(16, 1)
                else:
                    big_data, small_data = Variable(big_data.float()), Variable(small_data.float())
                    labels = Variable(labels).view(16, 1)

                inputs = torch.cat((big_data, small_data), 0)
                # zero the parameter gradients
                optimizer.zero_grad()

                # outputs = model(inputs)
                if phase == 'train':
                    outputs, aux_outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if phase == 'train':
                    loss = criterion(outputs[0:16], outputs[16:], labels)
                    loss.backward()
                    optimizer.step()
                else:
                    loss = criterion(outputs[0:16], outputs[16:], labels)
                outputs = outputs.view(2, 16)
                outputs = torch.transpose(outputs, 0, 1)

                _, preds = torch.min(outputs, 1)

                preds = preds.float().view(16, 1)
                # statistics
                running_loss += loss.data[0]
                running_corrects += (torch.sum(preds.float() == labels).data[0])

            epoch_loss = running_loss/32000    # dataset_sizes
            epoch_acc = running_corrects/32000     # dataset_sizes    # dataset_sizes

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
                torch.save(model.state_dict(), '../ckpt/rank_' + str(ckpt_index) + '.pth')
                ckpt_index += 1
                best_model_wts = model.state_dict()
                print('Best val Acc: {:4f}'.format(best_acc))
                result.writelines('\nBest val Acc: {:4f}'.format(best_acc))
        time_elapsed = time.time() - since
        print('epoch ' + str(epoch) + ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result.writelines('\nepoch ' + str(epoch) + ' training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                                                                   time_elapsed % 60))

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

    data_dir = '/home/jiaojiao/patch/One-Shot/rank'
    dataset = TargetDataset(data_dir, big_transform, small_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    global dataset_sizes
    dataset_sizes = len(dataset)

    model = torchvision.models.inception_v3(pretrained=False, num_classes=1) # batch_size = 16
    # model = torchvision.models.vgg19_bn(pretrained=False, num_classes=2) # batch_size = 10
    # model = torchvision.models.resnet50(pretrained=False, num_classes=2)  # batch_size = 50:12  101:8

    """
    model.load_state_dict(torch.load('./camelyon/Figure_6.pth'))
    model.AuxLogits.fc.weight.requires_grad = False
    model.AuxLogits.fc.bias.requires_grad = False
    model.fc.weight.requires_grad = False
    model.fc.bias.requires_grad = False
    """
    criterion = nn.MarginRankingLoss()

    if use_gpu:
        model.cuda()
        criterion.cuda()

    global result
    result = open('../ckpt/Figure_1.txt', 'a')
    result.writelines("use random seed: " + str(seed))
    lr = 0.0001
    print('lr = ' + str(lr))
    result.writelines('\nlr = ' + str(lr))
    weight_decay = 0.005
    print('weight_decay = ' + str(weight_decay))
    result.writelines('\nweight_decay = ' + str(weight_decay))
    result.flush()

    # Set SGD + Momentum
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.5, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=weight_decay)
    # Decay LR by a factor of 0.5 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=15)
    result.close()
