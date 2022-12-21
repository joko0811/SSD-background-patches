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
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        return image, image_path

    def __len__(self):
        return len(self.files)


class DirectoryTxtDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.files = sorted(glob.glob("%s/*.*" % self.image_path))
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.files[index % len(self.files)]

        with open(file_path) as f:
            s = f.read()

        if self.transform is not None:
            s = self.transform(s)
        return s, file_path

    def __len__(self):
        return len(self.files)
