import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.coco import CocoDetection

from tqdm import tqdm

from model import yolo
from dataset.simple import DirectoryDataset


def evaluation():

    gt_image_path = "./coco2014/images/train2014/"
    gt_annfile_path = "./coco2014/annotations/instances_train2014.json"

    adv_folder_path = "./testdata/evaluate/20221128_150124/"

    yolo_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((416, 416)),
    ])

    gt_set = CocoDetection(root=gt_image_path, annFile=gt_annfile_path,
                           transform=yolo_transforms)
    gt_loader = torch.utils.data.DataLoader(gt_set)

    adv_set = DirectoryDataset(
        image_path=adv_folder_path, transform=yolo_transforms)
    adv_loader = torch.utils.data.DataLoader(adv_set)

    set_size = len(adv_set)

    model = yolo.load_model(
        "weights/yolov3.cfg",
        "weights/yolov3.weights")
    model.eval()

    for (gt_image, _), adv_image in tqdm(zip(gt_loader, adv_loader), total=set_size):
        print("hoge")
    return


def main():

    evaluation()


if __name__ == "__main__":
    main()
