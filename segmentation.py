import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from annotations import get_scaled_annotations_PVOC, match_annotations
import os
from shutil import copyfile
import cv2
import pdb


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, image_folder, annotation_folder):
        self.images = [x for x in os.listdir(image_folder) if x.endswith(".png")]
        self.folder = image_folder
        self.to_tensor = torchvision.transforms.ToTensor()
        self.annotations = get_scaled_annotations_PVOC(annotation_folder)

    def __getitem__(self, idx):
        i_path = os.path.join(self.folder, self.images[idx])
        annotation = self.annotations[self.images[idx]]
        # annotation = self.to_tensor(annotation)
        img = Image.open(i_path)
        img = self.to_tensor(img)
        return img, annotation, self.images[idx]

    def __len__(self):
        return len(self.images)


def get_model(pretrained=True, num_classes=91, pretrained_backbone=True, device=None, mode="eval"):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained, num_classes=num_classes,
                                                               pretrained_backbone=pretrained_backbone)
    if device is not None:
        model.to(device)
    if mode == "train":
        model.train()
    else:
        model.eval()

    return model


def get_masks(image_folder, annotation_folder, out_folder="output"):
    device = torch.device("cuda")
    to_PIL = torchvision.transforms.ToPILImage()

    model = get_model(mode="eval", device=device)

    dataset = ImageLoader(image_folder, annotation_folder)
    obj_count = 0
    for i in range(len(dataset)):
        print("processing file", i)
        img, annotation, filename = dataset[i]
        c_img = img.to(device)
        out = model([c_img])[0]

        boxes, labels, scores, masks = out["boxes"], out["labels"], out["scores"], out["masks"]
        masks = masks.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()

        objects = boxes.shape[0]
        for obj in range(objects):
            if labels[obj] == 1 and scores[obj] > 0.8:
                obj_count += 1

                # Check if the detected object is a player
                bbox = boxes[obj]
                match = match_annotations(bbox, annotation, overlap=0.5)
                folder = "player" if match else "distract"

                # Copy the image
                image_in_path = os.path.join(image_folder, filename)
                image_out_path = os.path.join(out_folder, folder, str(obj_count) + ".png")
                copyfile(image_in_path, image_out_path)

                # Save the mask
                mask = (masks[obj][0] >= 0.5) * 255
                mask_out_path = os.path.join(out_folder, folder, str(obj_count) + ".pbm")
                cv2.imwrite(mask_out_path, mask)


if __name__ == "__main__":
    image_folder = "/home/chrizandr/sports/detection_exp/annotated"
    annotation_folder = "/home/chrizandr/sports/detection_exp/annotations"
    out_folder = "/home/chrizandr/sports/segmentation/"
    get_masks(image_folder, annotation_folder, out_folder)
