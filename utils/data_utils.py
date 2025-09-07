import torch
from torchvision import datasets,transforms

import matplotlib.pyplot as plt
from PIL import Image

# Checking an image in dataset

def show_image(image_path):
    img=Image.open(image_path).convert("RGB")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


#loading data

def data_loader(data_dir="dataset",batch_size=32,img_size=224):

    train_transforms = transforms.Compose([

        transforms.Resize((img_size, img_size)),#uniform img size
        transforms.RandomHorizontalFlip(),  # small augmentation (better generalization)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]) #standardizing pixel value for each color channel
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transforms)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes