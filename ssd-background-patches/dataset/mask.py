import glob
from PIL import Image
from torch.utils.data import Dataset


class DirectoryImageWithMaskDataset(Dataset):
    def __init__(self, image_path, mask_image_path, transform=None):
        self.image_path = image_path
        self.mask_image_path = mask_image_path
        self.files = sorted(glob.glob("%s/*.*" % self.image_path))
        self.mask_files = sorted(glob.glob("%s/*.*" % self.mask_image_path))
        self.transform = transform
        # TODO: Check if the names of image and mask match

    def __getitem__(self, index):
        image_path = self.files[index % len(self.files)]
        image = Image.open(image_path)

        mask_image_path = self.mask_files[index % len(self.mask_files)]
        mask_image = Image.open(mask_image_path)

        if self.transform is not None:
            image = self.transform(image)
            mask_image = self.transform(mask_image)

        return image, mask_image, image_path, mask_image_path

    def __len__(self):
        return len(self.files)
