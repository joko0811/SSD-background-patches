import glob
from PIL import Image
from torch.utils.data import Dataset


class DirectoryImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.files = sorted(glob.glob("%s/*.*" % self.image_path))
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        # TODO:replase torchvision.io.read_image
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.files)
