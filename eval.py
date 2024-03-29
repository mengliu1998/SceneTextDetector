import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', '../../Dataset/MLT2019/MLT_test', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoint/', '')
tf.app.flags.DEFINE_string('output_dir', './output/1', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
tf.app.flags.DEFINE_bool('confidence', True, 'output confidence score if needed')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape
    resize_w = w
    resize_h = h

    # limit the max side

    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    ###
    #im = (im - np.mean(im)) / max(np.std(im), 1.0 / np.sqrt(100000))
    ###
    return im, (ratio_h, ratio_w)

def detect(score_map, geo_map, timer, score_map_thresh=0.6, box_thresh=0.15, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*2, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start
    if boxes.shape[0] == 0:
        return None, timer
    # here we filter some low score boxes by the average score map, this is different from the orginal paper

    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 2, 1)
        #mask[score_map[:]<0.5] = 0 
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def box_area(box):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (box[1,0] - box[0,0]) * (box[1,1] + box[0,1]),
        (box[2,0] - box[1,0]) * (box[2,1] + box[1,1]),
        (box[3,0] - box[2,0]) * (box[3,1] + box[2,1]),
        (box[0,0] - box[3,0]) * (box[0,1] + box[3,1])
    ]
    return np.sum(edge)/2.

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        f_score, f_geometry = model.model(input_images, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            im_fn_list = get_images()
            total_time = 0
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                if boxes is not None:
                    confi = boxes[:, 8]
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                duration = timer['net']+timer['restore']+timer['nms']
                total_time += duration
                print('[timing] {}'.format(duration))
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                # save to file
                res_file = os.path.join(FLAGS.output_dir,'res_{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                with open(res_file, 'w') as f:
                    if boxes is not None:
                        count = -1
                        for box in boxes:
                            is_clockwise = True
                            count += 1
                            confidence = confi[count] 
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if box_area(box)>0:
                                is_clockwise = False
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            if FLAGS.confidence:
                                if is_clockwise:
                                    f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], confidence))
                                else:    
                                    f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[3,0], box[3,1], box[2,0], box[2,1],box[1, 0], box[1, 1], confidence))
                                    print('clockwise is corrected!')
                            else:
                                if is_clockwise:
                                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1],box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                                else:
                                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[3,0],box[3,1],box[2,0],box[2,1],box[1, 0], box[1, 1]))
                                    print('clockwise is corrected!')
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 0,255), thickness=3)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])
                    score_file = os.path.join(FLAGS.output_dir,'score_{}.jpg'.format(os.path.basename(im_fn).split('.')[0]))
                    my_score = np.squeeze(score,0)
                    cv2.imwrite(score_file, my_score*255)
            print('the average time for test: ',total_time/500)

if __name__ == '__main__':
    tf.app.run()
