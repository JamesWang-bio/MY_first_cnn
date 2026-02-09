
import torch
import main_control
import data_import

pixel_matrixs = (data_import.img_test_s, data_import.label_test)
imgs = data_import.img_test_s
labels = data_import.label_test

total = 0
correct = 0

test_num = 2000

'''==========================加载模型============================================================'''

# 创建模型实例
trained_model = main_control.my_module()  # 或者 main_control.MyModule()

# 加载状态字典（不需要 weights_only=True）
trained_model.load_state_dict(torch.load("D:/cnn_第一神经网络.pth"))
trained_model.eval()

for num in range(0, test_num):
    current_img = imgs[num]
    label = labels[num]

    with torch.no_grad():
        outputs = trained_model(current_img)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

#    print(label, predicted)

    if predicted == label:
        correct += 1


    total += 1

print(f"前{test_num}个样本的正确率：", correct/total)


