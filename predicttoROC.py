import torch
from AlexNet_new import AlexNet_improved
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt



label_map = {
    0:"cat",
    1:"dog",
}

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("./model/18 AlexNet_CatvsDog_improved.pth")
model.eval()
model.to(device)
img_dir = "./data/predict/"
img_files = os.listdir(img_dir)
for idx in range(len(img_files)):
    img_path = img_dir + img_files[idx]
  #  print(img_path)
    img = Image.open(img_path)
    img_data = data_transform(img)
    img_data = torch.unsqueeze(img_data, dim=0)

    predict = model(img_data.to(device))
   # print(predict)
    out = F.softmax(predict, dim=1)  # 修改这里的dim为1
    predict_class = torch.argmax(out, dim=1).cpu().numpy()  # 获取最可能的类别
    actual_probability = out[0][predict_class[0]].item()  # 获取实际概率
    print(f"Predicted class: {label_map[int(predict_class[0])]}, Probability: {actual_probability:.4f}")

   # print(out)
   # figure = plt.figure()

#    plt.title(label_map[int(predict)])
 #   plt.imshow(img)
  #  plt.show()