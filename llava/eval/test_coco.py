from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import os
import argparse

import json
from json import encoder
import random
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def process_result(r):
    r["caption"] = r["text"]
    return r

if __name__ == "__main__":

    # eval_dir = "/p/project/laionize/marianna/bakllava_original/BakLLaVA/llava/eval"

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--eval-dir", type=str)
    args = parser.parse_args()

    res_file = args.result_file
    eval_dir = args.eval_dir

    results = [process_result(json.loads(r)) for  r in open(res_file, "r")]
    # random.shuffle(results)
    # results = results[:1000]
    img_ids = [{"id":r["image_id"]} for r in results]

    res_file = os.path.join(eval_dir, "result_coco.json")


    with open(res_file, 'w') as f:
        json.dump(results, f)

    results = {r["question_id"]:r for r in results}



    ann_path = "/p/scratch/ccstdl/marianna/bakllava/eval_data/COCO2014/annotations/captions_val2014.json"
    with open(ann_path, "rb") as f:
        d = json.load(f)
    annotations = [a for a in d["annotations"] if a["id"] in results.keys()]

    d = {'annotations':annotations, 'images': img_ids}

    ann_file = os.path.join(eval_dir, "annotations.json")
    with open(ann_file, 'w') as f:
        json.dump(d, f)

    # res_file = "/p/project/laionize/marianna/bakllava_original/coco-caption/results/captions_val2014_fakecap_results.json"
    # ann_file = "/p/project/laionize/marianna/bakllava_original/coco-caption/annotations/captions_val2014.json"
    # print(d["annotations"][:10])

    # annFile = ""
    coco = COCO(ann_file)
    cocoRes = coco.loadRes(res_file)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()