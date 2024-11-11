import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ID2LABEL = {1: "印章", 2: "图片", 3: "标题", 4: "段落", 5: "表格", 6: "页眉", 7: "页码", 8: "页脚"}
ID2LABEL = {1: "印章", 2: "图片", 3: "标题", 4: "段落", 5: "表格", 6: "页眉", 7: "页码", 8: "页脚"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def main(args):
    base_path = Path(args.base_path)
    train_img_path = base_path / "train"
    val_img_path = base_path / "val"
    txts_path = base_path / "txts"

    output_base_path = Path(args.output_base_path)
    output_base_path.mkdir(exist_ok=True, parents=True)

    # create train.txt
    train_txt_path = output_base_path / "train.txt"
    with open(train_txt_path, "w") as f:
        f.write("\n".join(map(lambda x: "./images/" + x.name, train_img_path.iterdir())))

    # create val.txt
    val_txt_path = output_base_path / "val.txt"
    with open(val_txt_path, "w") as f:
        f.write("\n".join(map(lambda x: "./images/" + x.name, val_img_path.iterdir())))

    # create labels folder
    label_output_path = output_base_path / "labels"
    label_output_path.mkdir(exist_ok=True, parents=True)
    images_output_path = output_base_path / "images"
    images_output_path.mkdir(exist_ok=True, parents=True)

    for img_path in tqdm(list(train_img_path.iterdir()) + list(val_img_path.iterdir())):
        img = Image.open(img_path)
        img_width, img_height = img.size
        shutil.copy(img_path, images_output_path / img_path.name)

        txt_path = txts_path / (img_path.stem + ".txt")
        with open(txt_path, "r") as f:
            lines = f.readlines()

        labels = []
        for line in map(lambda x: x.split(","), lines):
            # p1p2p3p4 = np.array(line[:-1], dtype=np.float32).reshape(4, 2)
            p1p2p3p4 = np.array(line, dtype=np.float32).reshape(4, 2)
            # label_id = LABEL2ID[line[-1].strip()]
            label_id = 0
            p1p2p3p4 /= np.array([img_width, img_height])
            labels.append(f"{label_id} " + " ".join(map(str, p1p2p3p4.flatten().tolist())))

        with open(label_output_path / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(labels))

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="/workspace/datasets/table_rec/general_table_structure_cell_pad50_with_hsbc_ablation_line_exp11",
    )
    parser.add_argument(
        "--output_base_path",
        type=str,
        default="/workspace/datasets/table_rec/table_cell_det_yolo_format",
    )
    args = parser.parse_args()
    main(args)
