import glob
from PIL import Image
from torch.utils.data import Dataset


class DirectoryImageDataset(Dataset):
    def __init__(self, image_path, max_iter=None, transform=None):
        self.image_path = image_path
        self.files = sorted(glob.glob("%s/*.*" % self.image_path))
        if max_iter is not None:
            self.max_iter = min(max_iter, len(self.files))
            self.files = self.files[: self.max_iter]
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        # TODO:replase torchvision.io.read_image
        image = Image.open(image_path)

        image_size = (image.height, image.width)

        if self.transform is not None:
            image = self.transform(image)
        return image, image_size, image_path

    def __len__(self):
        return len(self.files)
