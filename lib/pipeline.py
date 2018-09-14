import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

import cv2
import time

from .helpers import box_iou
from .helpers import draw_text
from .helpers import draw_box_label


COLOR = {
    'car': (255, 0, 0),
    'bus': (0, 0, 255),
    'motorbike': (0, 255, 0),
    'bicycle': (255, 255, 0)
}


class Pipeline(object):
    def __init__(self, Tracker, detector=None, DEBUG=False):
        self.frame = 0
        self.max_age = 3
        self.min_hits = 2
        self.tracker_dict = {
            'car': [],
            'bus': [],
            'motorbike': [],
            'bicycle': []
        }
        self.counter = {
            'car': 0,
            'bus': 0,
            'motorbike': 0,
            'bicycle': 0
        }
        self.tracker_list = []
        self.detector = detector
        self.detector_is_set = False if detector is None else True
        self.Tracker = Tracker

        self.vidcap = None
        self.vidcap_success = True

        self.DEBUG = DEBUG

    def reset(self, Tracker=None, detector=None):
        Tracker = self.Tracker if Tracker is None else Tracker
        detector = self.detector if detector is None else detector
        self.__init__(Tracker, detector)

    def next_iteration(self, image, log_dir='./logs/result/', log_path='./log.txt', predicted=None):
        if not self.detector_is_set:
            raise ValueError(
                "Detector must be set first before running pipeline")

        img = np.copy(image)
        self.frame += 1
        if predicted is None:
            predicted = self.detector.get_localization(
                img
            )  # make sure get_localization response is {label: [[top, left, bottom, right],...] }

        if self.DEBUG:
            with open(log_path, "a") as myfile:
                myfile.write("FRAME: {}\n".format(str(self.frame)))
            debug_img = np.copy(img)
            for key in predicted:
                for ii, p_box in enumerate(predicted[key]['boxes']):
                    debug_img = draw_box_label(
                        debug_img, p_box.astype(int), label=key[0:2] +':{0:.2f}'.format(float(predicted[key]['scores'][ii])), box_color=(255,255,255), show_label=True)

        for key in self.tracker_dict:
            tracker_list = self.tracker_dict[key]
            z_box = predicted[key]['boxes']
            x_box = []
            for trk in tracker_list:
                x_box.append(trk.box)

            matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(
                x_box, z_box, iou_thrd=0.2)

            if self.DEBUG:
                with open(log_path, "a") as myfile:
#                    a = "FRAME: {}\n".format(str(self.frame))
                    b = "\t{}:\n".format(key)
                    c = "\t\tmatched:{}\n".format(str(len(matched)))
                    d = "\t\tunmatched_dets:{}\n".format(str(len(unmatched_dets)))
                    e = "\t\tunmatched_trks:{}\n".format(str(len(unmatched_trks)))
                    myfile.write(b + c + d + e)

            # tracker hits
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box = xx
                tmp_trk.hits += 1
                # tmp_trk.no_losses = 0

            # unmatched detector
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = self.Tracker()  # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = id(tmp_trk)
                tracker_list.append(tmp_trk)
                x_box.append(xx)

            # unmatched tracker
            for trk_idx in unmatched_trks:
                tmp_trk = tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx = [xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                x_box[trk_idx] = xx

            good_tracker_list = []
            for trk in tracker_list:

                if ((trk.hits >= self.min_hits)
                        and (trk.no_losses <= self.max_age)):

                    if not trk.counted:
                        self.counter[key] += 1
                        trk.counted = True

                    good_tracker_list.append(trk)
                    x_cv2 = trk.box

                    img = draw_box_label(
                        img, x_cv2, label=key, box_color=COLOR[key], show_label=False)
                    if self.DEBUG:
                        debug_img = draw_box_label(
                            debug_img, x_cv2, label=key, box_color=COLOR[key], show_label=False)

            self.tracker_dict[key] = [
                x for x in tracker_list if x.no_losses <= self.max_age
            ]
        img = draw_text(img, self.frame, self.counter)
        if self.DEBUG:
            debug_img = draw_text(debug_img,self.frame, self.counter)
            cv2.imwrite(log_dir + '{}.jpg'.format(str(self.frame).zfill(5)), debug_img)
        return img

    def load_video(self, file_path):
        self.vidcap = cv2.VideoCapture(file_path)

    def process_next_frame(self, safe_dir=None):
        if self.vidcap is None:
            raise ValueError("VIDCAP NONE")
        elif not self.vidcap_success:
            raise ValueError("LAST READ FAIL")
        else:
            self.vidcap_success, image = self.vidcap.read()
            if self.vidcap_success:
                img = self.next_iteration(image)
                if safe_dir is not None:
                    cv2.imwrite(
                        safe_dir + 'frame_{}.jpg'.format(str(self.frame)), img)
            return img

    # TODO(what to do?)
    def process_video(self, file_path=None, safe_frame=None):
        if file_path is not None:
            self.load_video(file_path)
        while self.vidcap_success:
            self.process_next_frame(safe_frame=safe_frame)

    def _batch_processing(self, image, q):
        pass

    def process_batch(self, n_batch=36):
        images = []
        for i in range(n_batch):
            self.vidcap_success, image = self.vidcap.read()
            if self.vidcap_success:
                images.append(image)
            else:
                print("END OF VIDEO")

        results = []
        start_time = time.time()
        predictions, feat_extract_time, tidying_time = self.detector.get_batch_localization(images)
        for i, pred in enumerate(predictions):
            results.append(self.next_iteration(images[i], predicted=pred))
        iteration_time = time.time() - start_time
        return results, feat_extract_time, tidying_time, iteration_time


def assign_detections_to_trackers(trackers, detections, iou_thrd=0.3):
    IOU_mat = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            IOU_mat[t, d] = box_iou(trk, det)

    matched_idx = linear_assignment(-IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t, trk in enumerate(trackers):
        if (t not in matched_idx[:, 0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if (d not in matched_idx[:, 1]):
            unmatched_detections.append(d)

    matches = []

    for m in matched_idx:
        if (IOU_mat[m[0], m[1]] < iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(
        unmatched_trackers)
