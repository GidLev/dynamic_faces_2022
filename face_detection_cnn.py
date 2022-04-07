import os
from deepface.detectors import FaceDetector
from matplotlib import pyplot as plt
from cv2 import imread
import matplotlib.patches as patches
import logging
import sys
import matplotlib.colors as mcolors
import time
import numpy as np
import pickle
from shapely.geometry import Polygon

class FaceFinder:
    """Detect faces in movie frames"""
    backends_detect = ['mtcnn', 'retinaface', 'ssd', 'dlib']
    backends_find = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace',
                     'DeepID', 'Dlib', 'ArcFace', 'Ensemble']
    colors = list(mcolors.TABLEAU_COLORS.keys())

    def __init__(self, frames_template, freq_movie, freq_final):
        self.params = {}
        self.params['movie_frames_template'] = frames_template
        self.params['freq_movie'] = freq_movie
        self.params['freq_final'] = freq_final
        self.logger = self._set_logger()
        self.start_time = time.perf_counter()
        self.detect_run = False
        self.find_run = False


    @staticmethod
    def _set_logger():
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
        handler.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(handler)
        return logger

    def _get_characters(self):
        self.characters = []

    def _rect_exist(self, rect, rects):
        poly = Polygon([(rect[0], rect[1]), (rect[0]+rect[2], rect[1]),
                        (rect[0]+rect[2], rect[1]+rect[3]), (rect[0], rect[1]+rect[3])])
        intersection = False
        for sec_rect in rects:
            other_polygon = Polygon([(sec_rect[0], sec_rect[1]),
                                  (sec_rect[0]+sec_rect[2], sec_rect[1]),
                                  (sec_rect[0]+sec_rect[2], sec_rect[1]+sec_rect[3]),
                                  (sec_rect[0], sec_rect[1]+sec_rect[3])])
            dice_numerator = 2 * poly.intersection(other_polygon).area
            dice_denominator = poly.area + other_polygon.area
            if (dice_numerator/ dice_denominator) > 0.2:
                intersection = True
                break
        return intersection

    def _get_rect_area(self, rect):
        from shapely.geometry import Polygon
        poly = Polygon([(rect[0], rect[1]), (rect[0]+rect[2], rect[1]),
                        (rect[0]+rect[2], rect[1]+rect[3]), (rect[0], rect[1]+rect[3])])
        return poly.area

    def detect(self, save_path = None):
        '''Boundry box around a face appearance'''
        n_frames = len(os.listdir(os.path.dirname(
            self.params['movie_frames_template'])))
        self.frames_list = [self.params['movie_frames_template'].format(x) for x in range(
            n_frames) if os.path.isfile(self.params['movie_frames_template'].format(x))]
        self.n_frames = len(self.frames_list)
        self.detected = {'frame'+str(i): {'rects': [], 'faces': [],
                                          'detector': [], 'area': []}
                         for i in range(self.n_frames)}
        for detector_name in self.backends_detect:
            detector = FaceDetector.build_model(detector_name)
            for frame_i, frame in enumerate(self.frames_list):
                frame = imread(frame)
                self.img_area = np.prod(frame.shape)
                try:
                    detected_faces = FaceDetector.detect_faces(detector, detector_name, frame)
                except Exception as e:
                    self.logger.warning(e)
                    detected_faces = []
                self.logger.info('In frame {}, {} faces detected with the {} '
                                 'detector.'.format(frame_i, len(detected_faces), detector_name))
                for detected_face in detected_faces:
                    if not self._rect_exist(detected_face[1],
                                      self.detected['frame'+str(frame_i)]['rects']):
                        self.detected['frame'+str(frame_i)][
                            'faces'].append(detected_face[0])
                        self.detected['frame'+str(frame_i)][
                            'rects'].append(detected_face[1])
                        self.detected['frame'+str(frame_i)][
                            'detector'].append(detector_name)
                        self.detected['frame'+str(frame_i)][
                            'area'].append(self._get_rect_area(detected_face[1]))

        # plot detected faces:
        text_kwargs = dict(ha='center', va='center', fontsize=14, color='C1')
        for frame_i, frame in enumerate(self.frames_list):
            frame = imread(frame)
            fig, ax = plt.subplots()
            plt.imshow(frame)
            for rect_i, rect in enumerate(self.detected['frame'+str(frame_i)]['rects']):
                rect_shape= patches.Rectangle((rect[0], rect[1]), rect[2], rect[3],
                                              linewidth=1, edgecolor='r',
                                              facecolor='none')
                ax.add_patch(rect_shape)
                plt.text(rect[0], rect[1], '{} ({})'.format(rect_i, self.detected[
                    'frame'+str(frame_i)]['detector'][rect_i]), **text_kwargs)
            if not save_path == None:
                plt.savefig(save_path + '/frame' +
                            str(frame_i) + '_detected.png')
            plt.show()
        self.detect_run = True

    def get_n_faces(self):
        if not self.detect_run:
            raise Exception('Please run detect first.')
        n_faces = np.zeros((self.n_frames))
        for frame_i, frame in enumerate(self.frames_list):
            for rect_i, rect in enumerate(
                    self.detected['frame'+str(frame_i)]['rects']):
                n_faces[frame_i] += 1
        return n_faces

    def get_faces_area(self):
        if not self.detect_run:
            raise Exception('Please run detect first.')
        faces_area = np.zeros((self.n_frames))
        for frame_i, frame in enumerate(self.frames_list):
            for area_i, area in enumerate(
                    self.detected['frame'+str(frame_i)]['area']):
                faces_area[frame_i] += area
        return faces_area

# first run ffmpeg to convert the movie to static image frames, the command (2Hz):
# ffmpeg -i "bangbangyouredead_v2 corrected.mov" -vsync cfr -r 2 -f image2 "video-frame%05d.png"
# then store the image in [movie_frames_2hz_path]:
movie_frames_2hz_path = '/path/to/frames/image/dir'
movie_frames2hz_template = movie_frames_2hz_path + '/video-frame{:05d}.png'

face_finder = FaceFinder(frames_template=movie_frames2hz_template, freq_movie=2,
                         freq_final=0.405)
face_finder.detect(save_path=movie_frames_2hz_path + '/detected')

with open(movie_frames_2hz_path + '/face_finder.pkl', 'wb') as f:
    pickle.dump(face_finder,f)

np.save(movie_frames_2hz_path + '/n_faces.npy', face_finder.get_n_faces())
np.save(movie_frames_2hz_path + '/faces_area.npy', face_finder.get_faces_area())

print('Done extracting faces.')
