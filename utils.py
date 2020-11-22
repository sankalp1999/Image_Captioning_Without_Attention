import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt


def save_checkpoint(state, filename="./my_checkpoint2.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_image2(inp, index, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    
 
    if title is not None:
        plt.title(title)
    name  = 'showcase' + str(index) + '.jpg'
    plt.savefig(name, bbox_inches='tight')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
