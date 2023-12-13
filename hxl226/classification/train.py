import os
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging


from model_v2 import MobileNetV2
from model_s import MobileNet_s


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main():
    logger = get_logger('./exp.log')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    f_epoch = 0  # 0 means training from scratch
    batch_size = 16
    epochs = 120-f_epoch
    num_classes = 101
    image_size = 256

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                     transforms.RandomErasing()
                                     ]),
        "val": transforms.Compose([transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # data_root = "E:/"
    # image_path = os.path.join(data_root, "data_set", "flower_data")  #data set path
    image_path = os.path.join(data_root, "data_set", "caltech-101")
    # image_path = os.path.join(data_root, "data_set", "caltech-256")
    # image_path = os.path.join(data_root, "data_set", "caltech_20%")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])


    train_num = len(train_dataset)

    label_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in label_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 1])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])

    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    # net = MobileNetV2(num_classes=num_classes).to(device)
    net = MobileNet_s(num_classes=num_classes).to(device)

    log_dir = "./Mobilenets_e4_20%c256_120_16bc_adamWlr0.001-0.00001.pth"

    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW([{'params': net.parameters(), 'initial_lr': 0.001}], lr=0.001)
    # optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0001)
    lf = lambda x: ((1 + math.cos(x * math.pi / (epochs*492))) / 2) * (1 - 0.01) + 0.01
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=34560, T_mult=2, eta_min=0.00002,
    #                                                            last_epoch=-1)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16360, T_mult=2, eta_min=0.00002,
    #                                                            last_epoch=-1)
    if os.path.exists(log_dir):
        net.load_state_dict(torch.load(log_dir))
    else:
        start_epoch = 0
        print('there is not any saved model, training from scratch')

    best_acc = 0.0
    best_classes_acc = list(0 for i in range(num_classes))

    save_path = './Mobilenets_e4_20%c256_120_16bc_adamWlr0.001-0.00001.pth'
    train_steps = len(train_loader)
    Loss_list = []
    Accuracy_list = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} learning rate:{}".format(epoch + 1 + f_epoch,
                                                                                      epochs + f_epoch,
                                                                                      loss,
                                                                                      optimizer.param_groups[0]['lr'])

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        correct = list(0 for i in range(num_classes))
        total = list(0 for i in range(num_classes))
        classes_acc = list(0 for i in range(num_classes))


        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                ture_false = torch.eq(predict_y, val_labels.to(device))
                for label_idx in range(len(ture_false)):
                    if ture_false[label_idx]:
                        correct[val_labels[label_idx].item()] += 1
                    total[val_labels[label_idx].item()] += 1

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1 + f_epoch,
                                                           epochs + f_epoch)
        for i in range(num_classes):
            classes_acc[i] = correct[i] / total[i]
        val_accurate = acc / val_num
        train_loss = running_loss / train_steps

        Loss_list.append(train_loss)
        Accuracy_list.append(100 * val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1 + f_epoch, train_loss, val_accurate))
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}\t lr={:.5f}'.format(epoch + 1 + f_epoch, epochs + f_epoch,
                                                                                  train_loss,
                                                                                  val_accurate,
                                                                                  optimizer.param_groups[0]['lr']))
        if val_accurate > best_acc:
            best_classes_acc = classes_acc
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('train loss vs. epoches')
    plt.ylabel('train loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")
    print('best acc: %.3f' % best_acc)
    print('classes accuracy is:')
    for i in range(len(best_classes_acc)):
        print('class %d accuracy is %.3f' % (i, best_classes_acc[i]))
    print('Finished Training')


if __name__ == '__main__':
    main()
