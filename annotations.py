"""Evaluate results and find the reasons for errors."""
import cv2
import numpy as np
import os
import pdb
import xml.etree.ElementTree as ET


def get_scaled_annotations_PVOC(annotation_dir, new_size=None):
    """Read and scale annotations based on new image size."""
    files = os.listdir(annotation_dir)
    annotations = dict()
    for f in files:
        try:
            file = ET.parse(os.path.join(annotation_dir, f))
            root = file.getroot()
        except Exception:
            pdb.set_trace()

        as_ = root.findall("object")
        size = root.findall("size")
        if len(size) == 1:
            size = size[0]
            width = int(size.findall("width")[0].text)
            height = int(size.findall("height")[0].text)
        else:
            size = None
            width = 0
            height = 0

        for annotation in as_:
            bbox = annotation.findall("bndbox")[0]
            xmin = float(bbox.findall("xmin")[0].text)
            ymin = float(bbox.findall("ymin")[0].text)
            xmax = float(bbox.findall("xmax")[0].text)
            ymax = float(bbox.findall("ymax")[0].text)

            if new_size is not None:
                new_h, new_w = new_size
                new_h, new_w = float(new_h), float(new_w)
                ymin = float(ymin/(height/new_h))
                ymax = float(ymax/(height/new_h))
                xmin = float(xmin/(width/new_w))
                xmax = float(xmax/(width/new_w))

            name = f.replace(".xml", ".png")
            if name in annotations:
                annotations[name] = np.vstack((annotations[name], [xmin, ymin, xmax, ymax]))
            else:
                annotations[name] = np.array([xmin, ymin, xmax, ymax]).reshape(1, 4)
    return annotations


# def evaluate(output_dir, data_dir, annotation_dir, size, overlap=0.5, filter_str1="", filter_str2=""):
#     """Evaluate output for the ball data."""
#     files = os.listdir(data_dir)
#     aps = []
#     for f in files:
#         annotation = annotations[f]
#         if filter_str1 not in f or filter_str2 not in f:
#             continue
#         if f not in output_imgs:
#             aps.append(0)
#             # print(f, f not in output_imgs)
#         else:
#             target_cls = np.zeros(annotation.shape[0])
#             pred_cls = np.zeros(preds.shape[0])
#             tps = match_annotations(preds, annotation, overlap)
#             p, r, ap, f1, _ = utils.ap_per_class(tps, conf, pred_cls, target_cls)
#             aps.append(ap[0])
#     if len(aps) == 0:
#         return 0
#     mean_ap = sum(aps) / len(aps)
#     return mean_ap


def match_annotations(output, annotation, overlap):
    """Match the annotations and output in the image and find accuracy."""
    dist = np.sum((annotation - output) ** 2, axis=1)
    closest_box = annotation[np.argmin(dist), :]
    iou = bb_intersection_over_union(closest_box, output)
    match = iou >= overlap

    return match


def bb_intersection_over_union(boxA, boxB):
    """Find IOU between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def mark_detection(img_file, output, annotation, output_dir):
    """Mark detection in image."""
    img = cv2.imread(img_file)

    for x in output:
        cv2.rectangle(img, tuple(x[0:2]), tuple(x[2:]), (0, 0, 255), 2)
    for x in annotation:
        cv2.rectangle(img, tuple(x[0:2]), tuple(x[2:]), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, img_file.split("/")[-1]), img)


def convert_to_yolo_annot(annotation, img_size=(720, 402)):
    xmin, ymin, xmax, ymax = annotation
    w, h = img_size
    x, y = float(xmin + xmax) / 2, float(ymin + ymax) / 2
    x, y = x / w, y / h
    w_rel, h_rel = float(xmax - xmin) / w, float(ymax - ymin) / h
    assert x <= 1 and y <= 1 and w_rel <= 1 and h_rel <= 1
    return [x, y, w_rel, h_rel]


def to_yolo_format(image_folder, annotation_folder, output_folder, image_size=(1280, 720)):
    images = [x for x in os.listdir(image_folder) if x.endswith(".png")]
    image_keys = [x.split("_")[0] for x in images]
    annotations = get_scaled_annotations_PVOC(annotation_folder)
    for f in annotations:
        print("Processing file", f)
        annot = annotations[f]
        annot_key = f.strip(".png")
        files = [images[i].replace(".png", ".txt") for i, x in enumerate(image_keys) if x == annot_key]
        annot = [convert_to_yolo_annot(x, image_size) for x in annot]
        out_str = "\n".join(["0 " + " ".join([str(x) for x in a]) for a in annot])
        for out_file in files:
            out = open(os.path.join(output_dir, out_file), "w")
            out.write(out_str)
            out.close()


if __name__ == "__main__":
    data_dir = "/ssd_scratch/cvit/chrizandr/synth_data/images"
    annotation_dir = "/ssd_scratch/cvit/chrizandr/synth_data/annotations"
    output_dir = "/ssd_scratch/cvit/chrizandr/synth_data/labels"
    to_yolo_format(data_dir, annotation_dir, output_dir)
