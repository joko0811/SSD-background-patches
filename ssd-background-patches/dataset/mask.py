import glob
from PIL import Image
from torch.utils.data import Dataset


class DirectoryImageWithMaskDataset(Dataset):
    def __init__(self, face_path, mask_path, max_iter=None, transform=None):
        self.face_path = face_path
        self.mask_path = mask_path
        self.face_files = sorted(glob.glob("%s/*.*" % self.face_path))
        self.mask_files = sorted(glob.glob("%s/*.*" % self.mask_path))

        if max_iter is not None:
            self.max_iter = max_iter
            self.face_files = self.face_files[: self.max_iter]
            self.mask_files = self.mask_files[: self.max_iter]

        self.transform = transform
        # TODO: Check if the names of image and mask match

    def __getitem__(self, index):
        image_path = self.face_files[index % len(self.face_files)]
        image = Image.open(image_path)
        image_size = (image.height, image.width)

        mask_image_path = self.mask_files[index % len(self.mask_files)]
        mask_image = Image.open(mask_image_path)

        if self.transform is not None:
            image = self.transform(image)
            mask_image = self.transform(mask_image)

        image_info = {}
        image_info["image_path"] = image_path
        image_info["mask_image_path"] = mask_image_path
        image_info["image_size"] = image_size

        return image, mask_image, image_info

    def __len__(self):
        return len(self.face_files)
