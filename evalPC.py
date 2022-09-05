# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import random
import colorsys
import colorsys
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pdb

bboxOriginal = 'bbox_out.json'
bboxSinModificiar = 'bbox_outUltra.json'
bboxModidicado = 'bbox_out_yolov4416_mia.json'
evaljson = './instances_val2017.json'
bboxpred = []
def app_eval():
	with open(bboxSinModificiar) as json_file:
		data = json.load(json_file)
		cnt = 0
		for i in data:
			cnt=cnt+1
			id_image = i['image_id']
			#category_id = i['category_id']
			category_id = convert_coco_category(i['category_id']-1)
			bbox = i['bbox']		
			score =  float(i['score'])
			bbox = [np.round(b * 100) / 100 for b in bbox]
			bbox = [b.item() for b in bbox]
			score = np.round(score * 10000) / 10000
			score = score.item()
			
			bboxpred.append({'image_id': id_image, 'category_id': category_id, 'bbox': bbox , 'score': score})
			#eval_ops = runDPU()
	with open(bboxModidicado, 'w') as json_file:
		json.dump(bboxpred, json_file)

def convert_coco_category(category_id):
    '''
    convert continuous coco class id (0~79) to discontinuous coco category id
    '''
    if category_id >= 0 and category_id <= 10:
        category_id = category_id + 1
    elif category_id >= 11 and category_id <= 23:
        category_id = category_id + 2
    elif category_id >= 24 and category_id <= 25:
        category_id = category_id + 3
    elif category_id >= 26 and category_id <= 39:
        category_id = category_id + 5
    elif category_id >= 40 and category_id <= 59:
        category_id = category_id + 6
    elif category_id == 60:
        category_id = category_id + 7
    elif category_id == 61:
        category_id = category_id + 9
    elif category_id >= 62 and category_id <= 72:
        category_id = category_id + 10
    elif category_id >= 73 and category_id <= 79:
        category_id = category_id + 11
    else:
        raise ValueError('Invalid category id')
    return category_id

if __name__ == "__main__":
	#app_eval()
	print("Generating COCO mAP score using ground truth values in {}".format(evaljson))
	cocoGt = COCO(evaljson)
	cocoDt = cocoGt.loadRes(bboxModidicado)
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')  # running bbox evaluation
	cocoEval.evaluate()
	cocoEval.accumulate()
	cocoEval.summarize()
	print(cocoEval.stats)
	print("done")
