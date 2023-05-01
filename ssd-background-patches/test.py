import hydra
import torch
import numpy as np
from PIL import Image
from imageutil import imgdraw
from torchvision import transforms

from model.S3FD import s3fd
from model.s3fd_util import S3fdResize

from model.Retinaface.detect import load_model
from model.Retinaface.data import cfg_re50
from model.Retinaface.models.retinaface import RetinaFace
from model.Retinaface.layers.functions.prior_box import PriorBox
from model.Retinaface.utils.box_utils import decode
from model.Retinaface.utils.box_utils import nms

from model.dsfd_util import DsfdResize
from model.DSFD.conf.widerface import widerface_640
from model.DSFD.model.face_ssd import build_ssd


def s3fd_demo():

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    S3FD_FACE_TRANSFORMS = transforms.Compose([
        S3fdResize(),
        transforms.PILToTensor(),
    ])

    image = Image.open("data/s3fd_test.jpg")
    # 変換前の画像の座標に変換するための配列
    scale = torch.tensor(
        [image.width, image.height, image.width, image.height]).to(device=device, dtype=torch.float)

    # transform
    s3fd_image = (S3FD_FACE_TRANSFORMS(image)).to(
        device=device, dtype=torch.float).unsqueeze(0)

    def _load_model(path):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Select device for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = s3fd.build_s3fd('test', 2)
        model.load_state_dict(torch.load(path))

        return model.to(device)
    model = _load_model("weights/s3fd.pth")

    model.eval()

    thresh = 0.6
    with torch.no_grad():
        output = model(s3fd_image)
        extract_output = output[output[..., 0] >= thresh]

        score = extract_output[..., 0]
        box = extract_output[..., 1:]*scale
        anno_image = imgdraw.draw_boxes(image, box)

    anno_image.save("hoge.png")


def retina_demo():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    S3FD_FACE_TRANSFORMS = transforms.Compose([
        transforms.PILToTensor(),
    ])

    image = Image.open("data/s3fd_test.jpg")
    # 変換前の画像の座標に変換するための配列
    scale = torch.tensor(
        [image.width, image.height, image.width, image.height]).to(device=device, dtype=torch.float)

    # transform
    s3fd_image = (S3FD_FACE_TRANSFORMS(image)).to(
        device=device, dtype=torch.float).unsqueeze(0)

    def _load_model(path):
        net = RetinaFace(cfg=cfg_re50, phase='test')
        model = load_model(net, path, False)
        return model.to(device)

    model = _load_model("./weights/Resnet50_Final.pth")
    model.eval()

    thresh = 0.6
    with torch.no_grad():
        loc, conf, landms = model(s3fd_image)

        priorbox = PriorBox(cfg_re50, image_size=(840, 840))
        priors = priorbox.forward().to(device)
        boxes = decode(loc.data.squeeze(0), priors, cfg_re50['variance'])
        conf_scores = conf.squeeze(0)[:, 1]

        """
        ext_idx = torch.where(conf_scores > pre_thresh)[0]
        ext_conf = conf_scores[ext_idx].clone()
        ext_boxes = boxes[ext_idx].clone()
        """

        ext1_idx, _ = nms(boxes, conf_scores, overlap=0.4, top_k=750)
        ext1_conf = conf_scores[ext1_idx].clone()
        ext1_boxes = boxes[ext1_idx].clone()

        ext2_idx = torch.where(ext1_conf > thresh)
        ext2_conf = ext1_conf[ext2_idx].clone()
        ext2_boxes = ext1_boxes[ext2_idx].clone()

        box = ext2_boxes*scale
        anno_image = imgdraw.draw_boxes(image, box)
    anno_image.save("hoge.png")


def dsfd_demo():

    DSFD_TRANSFORMS = transforms.Compose([
        DsfdResize(),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((104, 117, 123), 1),
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cfg = widerface_640
    net = build_ssd('test', cfg['min_dim'])  # initialize SSD

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
    net.cuda()
    net.eval()

    image = Image.open('data/retina_test.jpg')

    max_im_shrink = ((2000.0*2000.0) / (image.width * image.height)) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink

    scale = torch.tensor(
        [image.width, image.height, image.width, image.height]).to(device=device, dtype=torch.float)

    dsfd_img = DSFD_TRANSFORMS(image).unsqueeze(0).to(device=device)
    flip_dsfd_img = torch.flip(dsfd_img, [3])

    score_list = list()
    pt_list = list()
    with torch.no_grad():
        data = net(dsfd_img)
        for i in range(data.size(1)):
            j = 0
            while data[0, i, j, 0] >= cfg['conf_thresh']:
                score = data[0, i, j, 0]
                pt = (data[0, i, j, 1:] * scale)

                score_list.append(score)
                pt_list.append(pt)
                j += 1

                # data = net(dsfd_img)[1]
                # ext_data = data[data[:, :, 0] >= cfg['conf_thresh']]
                # score = ext_data[..., 0]
                # box = ext_data[..., 1:]*scale
                # flip_data = net(flip_dsfd_img)[1]

                # anno_image = imgdraw.draw_boxes(image, box)
                # anno_image.save("hoge.png")
    score_list = torch.tensor(score_list)
    pt_list = torch.stack(pt_list)

    anno_image = imgdraw.draw_boxes(image, pt_list)
    anno_image.save("hoge.png")


@ hydra.main(version_base=None, config_path="../conf/dataset/", config_name="casia_train")
def hydra_test(cfg):
    return


def main():
    # hydra_test()
    # s3fd_demo()
    # retina_demo()
    dsfd_demo()


if __name__ == '__main__':
    main()
