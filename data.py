import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from utils import set_sum, select_random_data
def getdataloader():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])

    train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
                                             download=False, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10,
                                               shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                              shuffle=True, num_workers=0)
    return train_loader, test_loader, 50000
def getpartdata():
    BATCH_SIZE = 10
    # a = torch.load('new_diffenrent_percent_data/train_data_image40.pth')
    # b = torch.load('new_diffenrent_percent_data/train_data_image40.pth')
    a, b, total_number = set_sum()
    c = a.view(len(a), 3, 224, 224)
    torch_dataset = Data.TensorDataset(c, b)
    train_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                              shuffle=True, num_workers=0)
    return train_loader, test_loader, total_number
def get_random_data():
    BATCH_SIZE = 10
    train_image_random, train_label_random, number = select_random_data()
    c = train_image_random.view(len(train_image_random), 3, 224, 224)
    torch_dataset = Data.TensorDataset(c, train_label_random)
    train_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    data_transform = {
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                              shuffle=True, num_workers=0)
    return train_loader, test_loader, number

def getdata():
    random = 2  # 控制随机数
    train_loader, test_loader = getdataloader()
    img_all_test0 = torch.zeros(2, 3, 224, 224)
    img_all_test1 = torch.zeros(2, 3, 224, 224)
    img_all_test2 = torch.zeros(2, 3, 224, 224)
    img_all_test3 = torch.zeros(2, 3, 224, 224)
    img_all_test4 = torch.zeros(2, 3, 224, 224)
    img_all_test5 = torch.zeros(2, 3, 224, 224)
    img_all_test6 = torch.zeros(2, 3, 224, 224)
    img_all_test7 = torch.zeros(2, 3, 224, 224)
    img_all_test8 = torch.zeros(2, 3, 224, 224)
    img_all_test9 = torch.zeros(2, 3, 224, 224)
    test_data_iter = iter(test_loader)
    for i in range(random):
       test_image, test_label = test_data_iter.next()
    # print(train_label)
    i0 = 0
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0
    i6 = 0
    i7 = 0
    i8 = 0
    i9 = 0
    for i in range(100):
        if test_label[i] == 0:
            if i0 == 2:
                continue
            img_all_test0[i0] = test_image[i]
            i0 += 1
        if test_label[i] == 1:
            if i1 == 2:
                continue
            img_all_test1[i1] = test_image[i]
            i1 += 1
        if test_label[i] == 2:
            if i2 == 2:
                continue
            img_all_test2[i2] = test_image[i]
            i2 += 1
        if test_label[i] == 3:
            if i3 == 2:
                continue
            img_all_test3[i3] = test_image[i]
            i3 += 1
        if test_label[i] == 4:
            if i4 == 2:
                continue
            img_all_test4[i4] = test_image[i]
            i4 += 1
        if test_label[i] == 5:
            if i5 == 2:
                continue
            img_all_test5[i5] = test_image[i]
            i5 += 1
        if test_label[i] == 6:
            if i6 == 2:
                continue
            img_all_test6[i6] = test_image[i]
            i6 += 1
        if test_label[i] == 7:
            if i7 == 2:
                continue
            img_all_test7[i7] = test_image[i]
            i7 += 1
        if test_label[i] == 8:
            if i8 == 2:
                continue
            img_all_test8[i8] = test_image[i]
            i8 += 1

        if test_label[i] == 9:
            if i9 == 2:
                continue
            img_all_test9[i9] = test_image[i]
            i9 += 1
        if i1+i2+i3+i4+i5+i6+i7+i8+i9 == 20:
            break
    c = torch.cat((img_all_test0, img_all_test1, img_all_test2, img_all_test3, img_all_test4,
                  img_all_test5, img_all_test6, img_all_test7, img_all_test8, img_all_test9), dim=0)
    return c

if __name__ == "__main__":
    b = getdata()
    print(b.size())