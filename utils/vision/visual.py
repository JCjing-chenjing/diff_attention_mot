


import cv2
import numpy as np







color = dict(red=(0, 0, 255),
                 green=(0, 255, 0),
                 blue=(255, 0, 0),
                 cyan=(255, 255, 0),
                 yellow=(0, 255, 255),
                 magenta=(255, 0, 255),
                 white=(255, 255, 255),
                 black=(0, 0, 0),
                my_color=(255, 0, 0)
             )


def draw_img(img, trackid_lst, box_lst, score_lst=None,
              bbox_color='green',
              text_color='green',
              thickness=1,
              font_scale=0.5
              ):
    '''
    box_lst：是左上角与右下角坐标
    '''
    for j, trackid in enumerate(trackid_lst):
        x1, y1, x2, y2 = np.array(box_lst[j]).astype(np.int32)
        bbox_color_new = color[bbox_color]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color_new, thickness=thickness)

        text_color_new = color[text_color]
        if score_lst is not None:
            score = round(score_lst[j], 4)
            label_text = 'trackid:{} score:{}'.format(str(trackid), str(score))
        else:
            label_text='track_id:{}'.format(str(trackid))
        cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color_new)
    return img









def show_video(image,width=720,height=540):
    # image = cv2.imread(dst_path + "/" + filename)
    # image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow('show_image', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('show_image', width, height)
    cv2.imshow('show_image', image)
    cv2.waitKey(10)































