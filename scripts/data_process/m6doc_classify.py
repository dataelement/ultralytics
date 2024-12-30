import base64
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import AzureOpenAI
from pycocotools.coco import COCO
from tqdm import tqdm


#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_response(image_path, prompt):
    base64_image = encode_image(image_path)
    client = AzureOpenAI(
        azure_endpoint="https://qinrui.openai.azure.com/",
        api_key="b81cd58db7384829828793e3845f19e5",
        api_version="2024-02-01",
    )
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        response = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        response = 'H'
    return response


if __name__ == '__main__':
    prompt = """请帮我对这张图片进行分类，你只需要回复我分类结果的字母，不要回复任何其他内容。
    类别选项：
    A. 课本
    B. 试卷
    C. 报纸
    D. 杂志
    E. 论文
    F. 笔记
    G. 拍照文档
    H. 其他
    """
    # coco_data_path = '/workspace/datasets/layout/unsv2_layout/M6Doc'
    # for split in ['train', 'val']:
    #     new_annos = defaultdict(list)
    #     coco = COCO(os.path.join(coco_data_path, f'{split}.json'))
    #     for image_id in tqdm(coco.getImgIds()):
    #         image_info = coco.loadImgs(image_id)[0]
    #         image_path = os.path.join(coco_data_path, 'images', image_info['file_name'])
    #         response = get_response(image_path, prompt)
    #         print(response)
    #         coco.imgs[image_id]['doc_category'] = response
    #     new_annos['images'] = list(coco.imgs.values())
    #     new_annos['categories'] = list(coco.cats.values())
    #     new_annos['annotations'] = list(coco.anns.values())
    #     with open(Path(coco_data_path, f'm6doc_new_{split}.json'), 'w') as f:
    #         json.dump(new_annos, f, ensure_ascii=False)

    file_path = '/workspace/datasets/layout/unsv2_layout/M6Doc/m6doc_new_val.json'

    new_coco = dict()
    coco = COCO(file_path)
    for image_id in tqdm(coco.getImgIds()):
        image_info = coco.loadImgs(image_id)[0]
        if 'A' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'school_textbook'
        elif 'B' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'exam_paper'
        elif 'C' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'newspaper'
        elif 'D' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'magazine'
        elif 'E' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'paper_zh_cn'
        elif 'F' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'handwrite_note'
        elif 'G' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'photo_doc'
        elif 'H' in image_info['doc_category']:
            coco.imgs[image_id]['doc_category'] = 'other'
        else:
            raise ValueError(f'Unknown doc_category: {image_info["doc_category"]}')
    new_coco['images'] = list(coco.imgs.values())
    new_coco['categories'] = list(coco.cats.values())
    new_coco['annotations'] = list(coco.anns.values())
    output_path = Path(file_path).parent / 'post_m6doc_new_val.json'
    with open(output_path, 'w') as f:
        json.dump(new_coco, f, ensure_ascii=False)

    # src_json = '/workspace/datasets/layout/unsv2_layout/M6Doc/val.json.bak'
    # mid_json = '/workspace/datasets/layout/unsv2_layout/M6Doc/_val.json'
    # dst_json = '/workspace/datasets/layout/unsv2_layout/M6Doc/val.json'

    # with open(src_json, 'r') as f:
    #     src_coco = json.load(f)
    # with open(mid_json, 'r') as f:
    #     mid_coco = json.load(f)

    # src_coco['images'] = mid_coco['images']
    # with open(dst_json, 'w') as f:
    #     json.dump(src_coco, f, ensure_ascii=False)
