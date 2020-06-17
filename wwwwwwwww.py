
ann = {}
ann['iscrowd'] = 1

if ann.get('iscrowd', False):
    print(1)
else:
    print(12)



keep_ratio=True
keep_ratio=False



filename = 'images/test/000000000019.jpg'

import cv2
import mmcv
import numpy as np

results = {}
img = cv2.imread(filename)


sizes=[(1333, 800), (1333, 768), (1333, 736), (1333, 704), (1333, 672), (1333, 640)]
scale_idx = np.random.randint(len(sizes))
img_scale = sizes[scale_idx]

results['filename'] = filename
results['img'] = img
results['img_shape'] = img.shape
results['ori_shape'] = img.shape
results['scale'] = img_scale



img33 = np.copy(results['img'])

if keep_ratio:
    img22, scale_factor22 = mmcv.imrescale(
        results['img'], results['scale'], return_scale=True)

    ori_h, ori_w = img33.shape[:2]
    scale_x = float(img_scale[0]) / ori_w
    scale_y = float(img_scale[1]) / ori_h
    scale_factor = min(scale_x, scale_y)
    img33 = cv2.resize(img33, None, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
else:
    img22, w_scale, h_scale = mmcv.imresize(
        results['img'], results['scale'], return_scale=True)
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)

    ori_h, ori_w = img33.shape[:2]
    scale_x = float(img_scale[0]) / ori_w
    scale_y = float(img_scale[1]) / ori_h
    img33 = cv2.resize(img33, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)


aaaaa = np.mean((img22 - img33)**2)

print()


