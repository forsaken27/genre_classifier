import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from config import Config
c = Config()

#########################################################################################
#                               Data preprocessing for CNN                              #
#########################################################################################
# original image size: (288, 432, 4)
train_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')),        # convert from RGBA into RGB
    # transforms.Resize((224, 224)),                        # square image for efficiency, to use pretrained models
    transforms.ToTensor(),                                # converting into tensor
    # transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),  # normalizing
])
# creating lazy-loading Dataset objects using ImageFolder
train_dataset = datasets.ImageFolder(root='./Data/images_original/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='./Data/images_original/val', transform=train_transform)
test_dataset = datasets.ImageFolder(root='./Data/images_original/test', transform=train_transform)         
# creating data loaders
train_loader = DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, num_workers=4, pin_memory=True)


#########################################################################################
#                            Data preprocessing for RandomForest                        #
#########################################################################################








if __name__ == "__main__":
    # Print dataset info
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Get one batch from the dataloader
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Show first 6 images from the batch
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    for i in range(6):
        img = images[i].permute(1, 2, 0).numpy()
        class_name = train_dataset.classes[labels[i]]
        
        axes[i].imshow(img)
        axes[i].set_title(f'Genre: {class_name}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("dataset_samples.png")