import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

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

device = "cpu"

model = torch.load("./model/18 AlexNet_CatvsDog_improved.pth")
model.eval()
model.to(device)
img_dir = "./data/predict/"
img_files = os.listdir(img_dir)
for idx in range(len(img_files)):
    img_path = img_dir + img_files[idx]
    print(img_path)
    img = Image.open(img_path)
    img_data = data_transform(img)
    img_data = torch.unsqueeze(img_data,dim=0)

    predict = model(img_data.to(device))

    # out = F.softmax(predict,dim=0)
    predict = torch.argmax(predict).cpu().numpy()
    print(predict)

    # print(out)
    figure = plt.figure()

    plt.title(label_map[int(predict)])
    plt.imshow(img)
    plt.show()