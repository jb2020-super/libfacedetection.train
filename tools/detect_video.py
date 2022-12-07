import argparse

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector

g_width = 0
g_height = 0

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('video', help='input video')
    parser.add_argument(
        '--score_thresh', type=float, default=0.5, help='score threshold')
    parser.add_argument(
        '--nms_thresh', type=float, default=0.45, help='nms threshold')
    parser.add_argument('--out', type=str, default='./work_dirs/result.mp4')
    parser.add_argument('--keepsize', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    cfg.model.test_cfg.score_thr = args.score_thresh
    cfg.model.test_cfg.nms.iou_threshold = args.nms_thresh

    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = None
    model.eval()

    vc = cv2.VideoCapture(args.video)
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_cnt = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    size = (width, height)
    global g_width 
    g_width = width
    global g_height 
    g_height = height

    print(size)
    if args.keepsize == False: 
        size = cal_size(width, height)
    print(size)
    fps = vc.get(cv2.CAP_PROP_FPS)
    vw = cv2.VideoWriter(args.out, fourcc, fps, size)
    tm = cv2.TickMeter()

    
    cnt = 0
    while(True) :
        ret, frame = vc.read()        
        if ret == False:
            break
        if args.keepsize == False:            
            frame = cv2.resize(frame, cal_size(frame.shape[1], frame.shape[0]))
        
        tm.start()
        detect_image(frame, model, vw)
        tm.stop()
        cnt += 1
        print("{}/{}, avg time {:.2f}, fps {:.2f}".format(cnt, frame_cnt, tm.getAvgTimeMilli(), tm.getFPS()))
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

def cal_size(width, height):
    max_len = max(width, height)
    scale = 1.0
    if max_len > 600:
        scale = max_len / 600
    return (round(width/scale), round(height/scale))



def detect_image(image, model, writer):    

    det_img, det_scale = resize_img(image, 'AUTO')
    image_tensor = torch.from_numpy(det_img).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    img_metas = [{
        'img_shape': det_img.shape,
        'ori_shape': image.shape,
        'pad_shape': det_img.shape,
        'scale_factor': [det_scale for _ in range(4)]
    }]
    with torch.no_grad():
        result = model.simple_test(image_tensor, img_metas, rescale=True)
    assert len(result) == 1
    result = result[0][0]
    save(image, result, writer)

def save(img, bboxes, writer):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f'{bbox[4]:.2f}', (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    global g_width     
    global g_height 
    #writer.write(cv2.resize(img, (g_width, g_height)))
    writer.write(img)
    


def resize_img(img, mode):
    if mode == 'ORIGIN':
        det_img, det_scale = img, 1.
    elif mode == 'AUTO':
        assign_h = ((img.shape[0] - 1) & (-32)) + 32
        assign_w = ((img.shape[1] - 1) & (-32)) + 32
        det_img = np.zeros((assign_h, assign_w, 3), dtype=np.uint8)
        det_img[:img.shape[0], :img.shape[1], :] = img
        det_scale = 1.
    else:
        if mode == 'VGA':
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(',')))
        assert len(input_size) == 2
        x, y = max(input_size), min(input_size)
        if img.shape[1] > img.shape[0]:
            input_size = (x, y)
        else:
            input_size = (y, x)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


if __name__ == '__main__':
    main()
