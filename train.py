import torch
from torch import nn
from resnet18 import Resnet18  # 确保 Resnet18 类正确导入
from torchvision import transforms
import os
from tqdm import tqdm

from my_dataset import MyDataSet  # 导入自定义数据集类

# 更新数据预处理
data_transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomCrop(128, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])
    ]),
    "val": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])
    ])
}

def read_data(data_path):
    images_path = []
    images_label = []
    for class_dir in os.listdir(data_path):
        class_dir_path = os.path.join(data_path, class_dir)
        for img_file in os.listdir(class_dir_path):
            images_path.append(os.path.join(class_dir_path, img_file))
            images_label.append(class_dir)
    return images_path, images_label

train_images_path, train_images_label = read_data('./128_128dataset_Improved/128_128dataset_Improved/train')
val_images_path, val_images_label = read_data('./128_128dataset_Improved/128_128dataset_Improved/val')


train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform["train"])
val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = Resnet18(num_classes=2).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # 添加 L2 正则化
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    loop = tqdm(dataloader, desc=f'\033[97m[Train epoch {epoch}]', total=len(dataloader), leave=True)
    total_acc = 0
    total_loss = 0
    total_samples = 0

    for X, y in loop:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_fn(output, y)
        _, predicted = torch.max(output.data, 1)
        acc = (predicted == y).sum().item()
        total_acc += acc
        total_samples += y.size(0)
        total_loss += loss.item() * y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        average_loss = total_loss / total_samples
        average_acc = total_acc / total_samples
        loop.set_postfix(loss=average_loss, acc=average_acc)

    return average_acc



def val(dataloader, model, loss_fn, epoch):
    model.eval()
    loop = tqdm(dataloader, desc=f'\033[97m[Valid epoch {epoch}]', total=len(dataloader), leave=True)
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            _, predicted = torch.max(output.data, 1)
            acc = (predicted == y).sum().item() / y.size(0)

            total_loss += loss.item()
            total_acc += acc
            n += 1

            loop.set_postfix(loss=total_loss / n, acc=total_acc / n)

    return total_acc / n

epochs = 50
best_train_accuracy = 0.0
best_val_accuracy = 0.0
for epoch in range(epochs):
    train_accuracy = train(train_dataloader, model, loss_fn, optimizer, epoch)
    val_accuracy = val(test_dataloader, model, loss_fn, epoch)
    lr_scheduler.step()

    model_saved = False  # Flag to check if model was saved in this epoch

    # Check if the current train accuracy is the best and save the model if so
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        os.makedirs('save_model', exist_ok=True)
        torch.save(model.state_dict(), f'save_model/m7_resnet18_epoch{epoch}_valacc{val_accuracy:.4f}.pth')
        tqdm.write(f'Saving best val model')
        model_saved = True

    # Check if the current validation accuracy is the best and save the model if so
    if train_accuracy > best_train_accuracy and not model_saved:
        best_train_accuracy = train_accuracy
        os.makedirs('save_model', exist_ok=True)
        torch.save(model.state_dict(), f'save_model/m7_resnet18_epoch{epoch}_trainacc{train_accuracy:.4f}.pth')
        tqdm.write(f'Saving best train model')

print('Training complete.')

