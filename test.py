# import torch
# from torch import nn
# from resnet18 import Resnet18  # 确保 Resnet18 正确导入
# from PIL import Image
# from torchvision import transforms
# import os
# import time
# import pynvml
#
# def load_images_from_folder(folder):
#     images = []
#     labels = []
#     for label in ['Defective', 'Non defective']:  # 假设您的标签为缺陷和无缺陷
#         class_folder = os.path.join(folder, label)
#         for filename in os.listdir(class_folder):
#             if filename.endswith(".jpg"):
#                 img_path = os.path.join(class_folder, filename)
#                 images.append(img_path)
#                 labels.append(label)
#     return images, labels
#
# # 初始化 pynvml
# pynvml.nvmlInit()
# handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个 GPU
#
# # 设定设备
# device = "cuda" if torch.cuda.is_available() else 'cpu'
#
# # 加载和准备模型
# model = Resnet18(num_classes=2).to(device)  # 设置类别数为2
# model.load_state_dict(torch.load("./save_model/m6_resnet18_epoch20_trainacc0.7699.pth"))
# model.eval()
#
# # 数据预处理
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])  # 归一化
# ])
#
# # 加载测试数据集
# folder_path = "./128_128dataset/Test"  # 更新为您的文件夹路径
# # folder_path = "./128_128dataset_Improved/128_128dataset_Improved/val"  # 更新为您的文件夹路径
# images, labels = load_images_from_folder(folder_path)
#
# total_predictions, correct_predictions = 0, 0
# total_inference_time = 0.0
# total_power_consumption = 0.0
#
# # 进行预测
# for img_path, true_label in zip(images, labels):
#     img = Image.open(img_path).convert('RGB')
#     img = data_transform(img)
#     img = img.unsqueeze(0).to(device)
#
#     start_time = time.time()  # 开始计时
#     power_start = pynvml.nvmlDeviceGetPowerUsage(handle)  # 开始功耗
#
#     with torch.no_grad():
#         outputs = model(img)
#         probabilities = nn.functional.softmax(outputs, dim=1)
#         predicted = torch.argmax(probabilities, 1)
#         predicted_label = 'Defective' if predicted == 0 else 'Non defective'
#         predicted_probability = probabilities[0, predicted].item()
#
#     power_end = pynvml.nvmlDeviceGetPowerUsage(handle)  # 结束功耗
#     inference_time = time.time() - start_time  # 结束计时
#
#     # 计算功耗差异
#     power_consumption = (power_end - power_start) / 1000  # 转换为瓦特
#     total_power_consumption += power_consumption
#
#     # 累计统计
#     correct_predictions += (predicted_label == true_label)
#     total_predictions += 1
#     total_inference_time += inference_time
#
#     print(f"Image: {os.path.basename(img_path)}, True label: {true_label}, Predicted label: {predicted_label}, Probability: {predicted_probability:.10f}, Inference Time: {inference_time:.10f} seconds, Power Consumption: {power_consumption:.4f} W")
#
# # 输出总体预测的精准度和推理时间
# accuracy = correct_predictions / total_predictions
# average_inference_time = total_inference_time / total_predictions
# average_power_consumption = total_power_consumption / total_predictions
#
# print(f'Overall Accuracy: {accuracy:.3f}')
# print(f'Total Inference Time: {total_inference_time:.10f} seconds')
# print(f'Average Inference Time per Image: {average_inference_time:.10f} seconds')
# print(f'Total Power Consumption: {total_power_consumption:.10f} W')
# print(f'Average Power Consumption per Image: {average_power_consumption:.10f} W')
#
# # 关闭 pynvml
# pynvml.nvmlShutdown()
import torch
from torch import nn
from resnet18 import Resnet18  # 确保 Resnet18 正确导入
from PIL import Image
from torchvision import transforms
import os
import time
import pynvml
import threading

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in ['Defective', 'Non defective']:  # 假设您的标签为缺陷和无缺陷
        class_folder = os.path.join(folder, label)
        for filename in os.listdir(class_folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(class_folder, filename)
                images.append(img_path)
                labels.append(label)
    return images, labels

def power_sampling(power_samples, stop_event, interval=0.01):
    while not stop_event.is_set():
        power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)  # 转换为瓦特
        time.sleep(interval)

# 初始化 pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用第一个 GPU

# 设定设备
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 加载和准备模型
model = Resnet18(num_classes=2).to(device)  # 设置类别数为2
model.load_state_dict(torch.load("./save_model/m6_resnet18_epoch20_trainacc0.7699.pth"))
model.eval()

# 数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4315, 0.3989, 0.3650], [0.2250, 0.2176, 0.2111])  # 归一化
])

# 加载测试数据集
folder_path = "./128_128dataset/Test"  # 更新为您的文件夹路径
images, labels = load_images_from_folder(folder_path)

# 测试10次
num_tests = 10
accuracy_list = []
total_inference_time_list = []
average_inference_time_list = []
total_power_consumption_list = []
average_power_consumption_list = []

power_sample_interval = 0.01  # 每10毫秒采样一次

for test in range(num_tests):
    total_predictions, correct_predictions = 0, 0
    total_inference_time = 0.0
    power_samples = []
    stop_event = threading.Event()

    # 启动功率采样线程
    power_thread = threading.Thread(target=power_sampling, args=(power_samples, stop_event, power_sample_interval))
    power_thread.start()

    # 测量开始时间
    start_time = time.time()

    # 进行预测
    for img_path, true_label in zip(images, labels):
        img = Image.open(img_path).convert('RGB')
        img = data_transform(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            inference_start_time = time.time()
            outputs = model(img)
            probabilities = nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, 1)
            predicted_label = 'Defective' if predicted == 0 else 'Non defective'
            predicted_probability = probabilities[0, predicted].item()
            inference_end_time = time.time()

        # 累计统计
        correct_predictions += (predicted_label == true_label)
        total_predictions += 1
        total_inference_time += (inference_end_time - inference_start_time)

    # 测量结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 停止功率采样线程
    stop_event.set()
    power_thread.join()

    if power_samples:
        # 计算功耗
        average_power = sum(power_samples) / len(power_samples)
        total_power_consumption = average_power * elapsed_time
    else:
        average_power = 0
        total_power_consumption = 0

    # 计算并记录每次测试的结果
    accuracy = correct_predictions / total_predictions
    average_inference_time = total_inference_time / total_predictions
    average_power_consumption = total_power_consumption / total_predictions

    accuracy_list.append(accuracy)
    total_inference_time_list.append(total_inference_time)
    average_inference_time_list.append(average_inference_time)
    total_power_consumption_list.append(total_power_consumption)
    average_power_consumption_list.append(average_power_consumption)

# 计算10次测试的平均值
avg_accuracy = sum(accuracy_list) / num_tests
avg_total_inference_time = sum(total_inference_time_list) / num_tests
avg_average_inference_time = sum(average_inference_time_list) / num_tests
avg_total_power_consumption = sum(total_power_consumption_list) / num_tests
avg_average_power_consumption = sum(average_power_consumption_list) / num_tests

print(f'Average Overall Accuracy: {avg_accuracy:.3f}')
print(f'Average Total Inference Time: {avg_total_inference_time:.10f} seconds')
print(f'Average Inference Time per Image: {avg_average_inference_time:.10f} seconds')
print(f'Average Total Power Consumption: {avg_total_power_consumption:.10f} W')
print(f'Average Power Consumption per Image: {avg_average_power_consumption:.10f} W')

# 关闭 pynvml
pynvml.nvmlShutdown()
