import numpy as np
import cv2


def box_iou(a, b):
    w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0]) * (a[3] - a[1])
    s_b = (b[2] - b[0]) * (b[3] - b[1])

    return float(s_intsec) / (s_a + s_b - s_intsec)


def draw_text(img, frame, count_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_color = (255, 255, 255)
    left = 5
    top = 15
    text = "frame: " + str(frame)
    cv2.putText(img, text, (left, top), font, font_size, font_color, 1, cv2.LINE_AA)
    top += 12
    for key in count_dict:
        text = key + ": " + str(count_dict[key])
        cv2.putText(img, text, (left, top), font, font_size, font_color, 1,
                    cv2.LINE_AA)
        top += 12
    return img


def draw_box_label(img,
                   bbox_cv2,
                   label,
                   box_color=(0, 255, 255),
                   show_label=True):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.35
    font_color = (0, 0, 0)
    top, left, bottom, right = bbox_cv2
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 1)

    if show_label:
        cv2.rectangle(img, (left, top - 12), (right, top), box_color,
                      -1, 1)
        cv2.putText(img, label, (left, top - 8), font, font_size, font_color,
                    1, cv2.LINE_AA)

    return img
