import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from xnas.datasets.transforms import MultiSizeRandomCrop

msrc = MultiSizeRandomCrop([4,224])
# print(MultiSizeRandomCrop.CANDIDATE_SIZES)

for i in range(5):
    MultiSizeRandomCrop.sample_image_size()
    print(MultiSizeRandomCrop.ACTIVE_SIZE)

def my_collate(batch):
    msrc.sample_image_size()
    xs = torch.stack([i[0] for i in batch])
    ys = torch.Tensor([i[1] for i in batch])
    return [xs, ys]

T = transforms.Compose([
    msrc,
    transforms.ToTensor(),
])

_data = dset.CIFAR10(
    root='./data/cifar10',
    train=True,
    transform=T,
)

loader = data.DataLoader(
    dataset=_data,
    batch_size=128,
    collate_fn=my_collate,
)

for i, (trn_X, trn_y) in enumerate(loader):
    print(trn_X.shape, trn_y.shape)
    if i==5:
        break
