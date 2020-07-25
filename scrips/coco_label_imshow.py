import cv2
import os
import random
import json
from collections import defaultdict
#随机打印coco数据集图片以及gt框来检验数据集标注是否有问题 并计算统计特征
os.chdir(os.path.join('..', 'datasets', 'coco_shape'))
# category={1:'person',2:'bicycle',3:'car',4:'dog'}
# category = {
#     1: 'knife',
#     2: 'scissors',
#     3: 'lighter',
#     4: 'zippooil',
#     5: 'pressure',
#     6: 'slingshot',
#     7: 'handcuffs',
#     8: 'nailpolish',
#     9: 'powerbank',
#     10: 'firecrackers'
# }
category = {1: 'rectangle', 2: 'circle'}
# category={1:"Consolidation",
# 2:"Fibrosis",
# 3:"Effusion",
# 4:"Nodule",
# 5:"Mass",
# 6:"Emphysema",
# 7:"Calcification",
# 8:"Atelectasis",
# 9:"Fracture"
# }
# with open(os.path.join('instances_train2017.json'), 'r') as f:
with open(os.path.join('annotations', 'instances_train2017.json'), 'r') as f:
    label = json.load(f)
    images = label['images']
    categories = label['categories']
    annotations = label['annotations']
    random.shuffle(images)
    stat = defaultdict(int)
    for img in images:
        # for img in images[:200]:
        bbox = []
        cid = 0
        img_name = img['file_name']
        # if 'ar' not in img_name:continue
        img_id = img['id']
        flag = 0
        for res in annotations:  #这两个if先后顺序很重要 倒换的话第二个if得变成elif
            if res['image_id'] != img_id and flag > 0: break
            if res['image_id'] == img_id:
                flag += 1
                bbox.append([*res['bbox'], res['category_id']])
        pic = cv2.imread(os.path.join('train2017', img_name))
        h_, w_, c_ = pic.shape
        for box in bbox:
            x, y, w, h, id = list(map(float, box))
            stat[id] += 1
            # x, y, w, h = x * w_, y * h_, w * w_, h * h_
            x, y, w, h, id = list(map(int, [x, y, w, h, id]))
            cv2.rectangle(pic, (x, y), (x + w, y + h), (0, 255, 0), 2)  #2是线的宽度
            cv2.putText(pic, '{}, {:.3f}'.format(category[id], 2), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 0), 2)
        cv2.imwrite(os.path.join('/home','lh','myhome','Yolov3_tricks','images','{}_visual.jpg'.format(img_name.split('.')[0])), pic)
        # cv2.namedWindow(img_name, 0)
        # cv2.resizeWindow(img_name, 416, 416)
    #     cv2.imshow(img_name, pic)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
print(stat)
# x-ray比赛的数据 统计信息

# def lh_defaultdict():
#     return defaultdict(list)

# stat = defaultdict(lh_defaultdict)
# for i in category.values():
#     stat[i]['width'] = [10000, 0]
#     stat[i]['height'] = [10000, 0]
#     stat[i]['aspect'] = [10000, 0]
#     stat[i]['loc'] = [10000,0,10000,0]

# with open(os.path.join("train_x-ray.json"), 'r') as f:
#     label = json.load(f)
#     for i in range(len(label)):
#         if label[i]['boxes'] == [] or ('Nodule' not in label[i]['syms'] ):
#             continue
#         else:
#             img_name = label[i]['file_name']
#             pic = cv2.imread(os.path.join('val', img_name),0 )
#             for j in range(len(label[i]["syms"])):
#                 if label[i]['syms'][j] not in ['Nodule']:continue
#                 x1, y1, x2, y2 = label[i]['boxes'][j]

#                 bbox_name = label[i]['syms'][j]
#                 w, h = x2 - x1, y2 - y1
# #                 stat[bbox_name]['width'][0] = min(stat[bbox_name]['width'][0], w)
# #                 stat[bbox_name]['width'][1] = max(stat[bbox_name]['width'][1], w)
# #                 stat[bbox_name]['height'][0] = min(stat[bbox_name]['height'][0], h)
# #                 stat[bbox_name]['height'][1] = max(stat[bbox_name]['height'][1], h)
# #                 stat[bbox_name]['aspect'][0] = min(stat[bbox_name]['aspect'][0], w/h)
# #                 stat[bbox_name]['aspect'][1] = max(stat[bbox_name]['aspect'][1], w/h)
# #                 stat[bbox_name]['loc'][0] = min(stat[bbox_name]['loc'][0], y1)
# #                 stat[bbox_name]['loc'][1] = max(stat[bbox_name]['loc'][1], y2)
# #                 stat[bbox_name]['loc'][2] = min(stat[bbox_name]['loc'][2], x1)
# #                 stat[bbox_name]['loc'][3] = max(stat[bbox_name]['loc'][3], x2)

# # print(stat)
# # with open(r'D:\Ubuntu_Server_Share\lh_CODE\stat.txt', 'w') as f:
# #     json.dump(stat,f)

#                 cv2.rectangle(pic, (x1, y1), (x2 , y2), (0, 255, 0), 2)  #2是线的宽度
#                 cv2.putText(pic, '{}, {:.3f}'.format(label[i]['syms'][j], 0.5),
#                             (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                             (255, 255, 0), 2)
#             cv2.namedWindow(img_name,0)
#             cv2.resizeWindow(img_name, 600, 600)
#             # pic = cv2.resize(pic, 416, 416)
#             cv2.imshow(img_name, pic)
#             cv2.waitKey(0)
# cv2.destroyAllWindows()
