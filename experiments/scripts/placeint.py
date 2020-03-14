from __future__ import absolute_import, division, print_function

# import some common libraries
import os
from os import path as osp
import time
import yaml

import sys
from sys import stdout
import cv2
from datetime import datetime, timedelta
import json
import itertools
import logging
import shutil
import time

import pytz
import psycopg2
import PIL
import pandas as pd
import imageio
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
import hashlib

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='latest.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logger.handlers[0].stream = sys.stdout

import numpy as np
import torch
from torch.utils.data import DataLoader
from sacred import Experiment

# utility function
from helpers.db_opt import connect_to_db
from helpers.utils import *

from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.resnet import resnet50
from tracktor.tracker import Tracker
from tracktor.utils import interpolate, plot_sequence

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_network_config'])
ex.add_config(ex.configurations[0]._conf['tracktor']['obj_detect_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

# Tracker = ex.capture(Tracker, prefix='tracker.tracker')
class detection():

    def __init__(self, video_filename, config_path, client_id, ts_start, floor_id, camera_id):

        with open(config_path) as jsonfile:
            self.config = json.load(jsonfile)

        # basic information on the video
        self.video_filename = video_filename
        self.video_path        = os.path.join('..', 'data', 'input', video_filename)
        self.video_id        = video_filename.split('.')[0]
        self.detect_per_sec = self.config['general']['detect_per_sec']
        self.client_id        = client_id
        self.floor_id       = floor_id
        self.camera_id        = camera_id
        self.PROCESS_GAZE_ESTIMATION = self.config['general']['PROCESS_GAZE_ESTIMATION']

        # see if video exists
        if not os.path.exists(self.video_path):
            logger.error("video not found")
            assert 0, "video not found"

        
        # get cv2 video capture
        self.cap = cv2.VideoCapture('../data/input/%s' % video_filename)
        # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        # self.duration = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
        _, sample = self.cap.read()
        self.w = sample.shape[1]
        self.h = sample.shape[0]
        # self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
        self.engine = create_engine('''postgresql://%s:%s@%s:%s/%s''' % (self.config['db']['user'], 
                                                              "placeint", self.config['db']['host'], str(5432),
                                                              "placeint"))

        self.frame_count = int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.duration = self.frame_count / self.fps 
        logger.info("fps %s, frame count %s, duration %s" % (str(self.fps), str(self.frame_count), str(self.duration)))

        self.frames_per_detect = int(int(self.fps) / int(self.detect_per_sec))

        # start end time of the video
        self.ts_start = ts_start
        self.ts_end   = self.ts_start + timedelta(0, self.duration)
        logger.info("time start %s" % str(self.ts_start))
        logger.info("time end %s" % str(self.ts_end))
        
        # current time
        now = pytz.utc.localize(datetime.now())
        self.created_at = now.strftime("%Y-%m-%d %H:%M:%S %Z")
        logger.info("created at %s" % str(self.created_at))

        # =================== to-do add export to video code and corresponding config keys
        # self.out_segm = imageio.get_writer('../data/output/%s_segm.mp4' % self.video_id, fps=self.fps)
        # self.out_head = imageio.get_writer('../data/output/%s_head.mp4' % self.video_id, fps=self.fps)
        self.out_peds = imageio.get_writer('../data/output/%s_peds.mp4' % self.video_id, fps=self.fps)
        # self.out_uv   = imageio.get_writer('../data/output/%s_uv.mp4' % self.video_id, fps=self.fps)
        # self.out_video_dir = os.path.join('../data/output', video_id + '.mp4')

        self.ped_img_dir = os.path.join('../data/pedestrians_images', self.video_id)
        # self.uv_img_dir = os.path.join('../data/reference_files/uv', video_id)
        # self.segm_img_dir = os.path.join('../data/reference_files/segm', video_id)
        # self.head_bbx_dir = os.path.join('../data/reference_files/head', video_id)
        # self.peds_bbx_dir = os.path.join('../data/reference_files/peds', video_id)
        # self.screenshot_dir = os.path.join('../data/reference_files/screenshots', video_id)

        # create_folder(self.uv_img_dir)
        # create_folder(self.segm_img_dir)
        # create_folder(self.head_bbx_dir)
        # create_folder(self.peds_bbx_dir)
        # create_folder(self.screenshot_dir)
        
        self.con, self.cur = connect_to_db(user=self.config['db']['user'], pwd="placeint", 
                        host=self.config['db']['host'], 
                        port=5432, db="placeint")


    def check_process_has_row(self):
        # check if the row exists
        query = '''SELECT COUNT(*) FROM public.processes WHERE filename = '%s' ''' % (self.video_filename)
        self.cur.execute(query)
        self.con.commit()

        if self.cur.fetchone()[0] == 0:
            print('adding row to processing table')
            insert_record = pd.DataFrame({'filename': self.video_filename, 'client_id': self.client_id, 'ts_start': self.ts_start, 
                            'duration': int(self.duration), 'step_0_detecting_and_tracking': None, 
                            'step_1_gaze_estimation': None, 'step_2_classification': None,
                            'step_3_kpi': None}, index=[0])
            insert_record.to_sql('processes', self.engine, if_exists="append", index=False)
        else:
            print('row already exist in processing table')

    def check_if_run(self):
        # check if step 0 is already processed
        query = '''SELECT COUNT(*) FROM public.processes WHERE filename = '%s' AND %s IS NOT NULL''' % (self.video_filename, 'step_0_detecting_and_tracking')
        self.cur.execute(query)
        self.con.commit()
        if self.cur.fetchone()[0] >= 1:
            return 0
        else:
            return 1
        
    def update_process_table(self):
        query = '''UPDATE public.processes SET %s = current_timestamp WHERE filename = '%s' ''' % ('step_0_detecting_and_tracking', self.video_filename)
        self.cur.execute(query)
        self.con.commit()

    @ex.automain
    def run(tracktor, siamese, _config):

        self.tracking_df = pd.DataFrame(columns=['frame', 'time', 'identity', 'peds_bbx', 'head_bbx'])
        run_count = 0
        
        # array for storing the quality score for each person at each frame
        image_quality_for_each_person = {}
        
        # empty output folder
        if os.path.isdir(os.path.join('../data/pedestrians_images', self.video_id)):
            shutil.rmtree(os.path.join('../data/pedestrians_images', self.video_id))

        logger.info('start to track file %s' % self.video_id)
        
        # -------------
        
        # set all seeds
        torch.manual_seed(tracktor['seed'])
        torch.cuda.manual_seed(tracktor['seed'])
        np.random.seed(tracktor['seed'])
        torch.backends.cudnn.deterministic = True

        output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
        sacred_config = osp.join(output_dir, 'sacred_config.yaml')

        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        with open(sacred_config, 'w') as outfile:
            yaml.dump(_config, outfile, default_flow_style=False)

        ##########################
        # Initialize the modules #
        ##########################

        # object detection
        print("[*] Building object detector")
        if tracktor['network'].startswith('frcnn'):
            # FRCNN
            from tracktor.frcnn import FRCNN
            from frcnn.model import config

            if _config['frcnn']['cfg_file']:
                config.cfg_from_file(_config['frcnn']['cfg_file'])
            if _config['frcnn']['set_cfgs']:
                config.cfg_from_list(_config['frcnn']['set_cfgs'])

            obj_detect = FRCNN(num_layers=101)
            obj_detect.create_architecture(2, tag='default',
                anchor_scales=config.cfg.ANCHOR_SCALES,
                anchor_ratios=config.cfg.ANCHOR_RATIOS)
            obj_detect.load_state_dict(torch.load(tracktor['obj_detect_weights']))
            
        elif tracktor['network'].startswith('fpn'):
            # FPN
            from tracktor.fpn import FPN
            from fpn.model.utils import config
            config.cfg.TRAIN.USE_FLIPPED = False
            config.cfg.CUDA = True
            config.cfg.TRAIN.USE_FLIPPED = False
            checkpoint = torch.load(tracktor['obj_detect_weights'])

            if 'pooling_mode' in checkpoint.keys():
                config.cfg.POOLING_MODE = checkpoint['pooling_mode']

            set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                        'ANCHOR_RATIOS', '[0.5,1,2]']
            config.cfg_from_file(_config['tracktor']['obj_detect_config'])
            config.cfg_from_list(set_cfgs)

            obj_detect = FPN(('__background__', 'pedestrian'), 101, pretrained=False)
            obj_detect.create_architecture()

            obj_detect.load_state_dict(checkpoint['model'])
        else:
            raise NotImplementedError(f"Object detector type not known: {tracktor['network']}")

        obj_detect.eval()
        obj_detect.cuda()

        # reid
        reid_network = resnet50(pretrained=False, **siamese['cnn'])
        reid_network.load_state_dict(torch.load(tracktor['reid_network_weights']))
        reid_network.eval()
        reid_network.cuda()

        # tracktor
        if 'oracle' in tracktor:
            tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
        else:
            tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

        print("[*] Beginning evaluation...")

        time_total = 0
        for sequence in Datasets(tracktor['dataset']):
            tracker.reset()

            now = time.time()

            print("[*] Evaluating: {}".format(sequence))

            data_loader = DataLoader(sequence, batch_size=1, shuffle=False)
            for i, frame in enumerate(data_loader):
                if len(sequence) * tracktor['frame_split'][0] <= i <= len(sequence) * tracktor['frame_split'][1]:
                    tracker.step(frame)
            results = tracker.get_results()

            time_total += time.time() - now

            print("[*] Tracks found: {}".format(len(results)))
            print("[*] Time needed for {} evaluation: {:.3f} s".format(sequence, time.time() - now))

            if tracktor['interpolate']:
                results = interpolate(results)

            sequence.write_results(results, osp.join(output_dir))

            if tracktor['write_images']:
                plot_sequence(results, sequence, osp.join(output_dir, tracktor['dataset'], str(sequence)))

        print("[*] Evaluation for all sets (without image generation): {:.3f} s".format(time_total))

    def run(self):

        start_time_0 = time.time()
        for k in range(self.frame_count):

            if k > 10000:
                break

            valid, image = self.cap.read()

            # check if reached end of video file
            if valid == False:
                break

            # skip those frames based on detection per second 
            if not k % self.fps % self.frames_per_detect == 0:
                continue
        
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # for imageio
            instances = predictor(image)["instances"].to("cpu")
            self.ins = instances

            cls_ids = instances.pred_classes

            # filter bbx by prediction class
            bbox = np.array([[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for i, 
                             box in enumerate(instances.pred_boxes.tensor) if cls_ids[i] == 0])
            
            cls_conf = instances.scores
            # filter cls_conf by bbox
            cls_conf = [conf for i, conf in enumerate(cls_conf) if cls_ids[i] == 0]

            stdout.write("\rtotal frame count is %s" % (str(k)))
            stdout.flush()

            peds_bbx_vis = image.copy()

            if not len(bbox) == 0:

                if self.PROCESS_GAZE_ESTIMATION:
                    bbx_head, bbx_mask = self.extract_head(segm_extractor, segm_visualizer, instances, self.w, self.h)
                else:
                    bbx_head, bbx_mask = ([None for i in bbox], [None for i in bbox])

                # to-do: using only human image for deep sort
                if self.config["general"]["tracker"] == "deepsort":
                    outputs, bbx_head = tracker.update(bbox, bbx_head, cls_conf, image)
                elif self.config["general"]["tracker"] == "tracktor":
                    tracker.step([image], bbox, cls_conf)
                    outputs = tracker.get_results()
                    self.check000 = tracker
                    continue
                # check if output of tracker is > 0
                if len(outputs) > 0:
                    if self.config["general"]["tracker"] == "deepsort":
                        bbx_xyxy = outputs[:,:4]
                        identities = outputs[:,4]
                    elif self.config["general"]["tracker"] == "tracktor":
                        tracker.step([image], bbox, cls_conf)
                        outputs = tracker.get_results()

                    identities = [(hashlib.md5((str(i) + str(self.ts_start) + 
                                                self.camera_id).encode('utf-8'))).hexdigest() for i in identities]

                    # compare and keep cropped images for human, for future attribute classification
                    for i, b in enumerate(outputs):
                        
                        bbx_xyxy = [[int(round(a[0])), int(round(a[1])), 
                                     int(round(a[2])), int(round(a[3]))] for a in bbx_xyxy]
                        
                        # skip saving the image if x1 == x0 or y0 == y1, which sometimes happens
                        if (bbx_xyxy[i][1] == bbx_xyxy[i][3]) or (bbx_xyxy[i][0] == bbx_xyxy[i][2]):
                            continue
                            
                        # crop pedestrian images
                        # using only human image for deep sort
                        # crop_img = only_human_image[bbx_xyxy[i][1]:bbx_xyxy[i][3],
                        #             bbx_xyxy[i][0]:bbx_xyxy[i][2]]
                        crop_img = image[bbx_xyxy[i][1]:bbx_xyxy[i][3],
                                    bbx_xyxy[i][0]:bbx_xyxy[i][2]]

                        
                        # calculate quality score by cropped image size and ratio, the best h/w ratio is 1.7 for now
                        ratio = (bbx_xyxy[i][3] - bbx_xyxy[i][1]) / (bbx_xyxy[i][2] - bbx_xyxy[i][0])
                        size = (bbx_xyxy[i][3] - bbx_xyxy[i][1]) * (bbx_xyxy[i][2] - bbx_xyxy[i][0])
                        quality = size / (abs(ratio - 1.7) + 0.5)

                        if identities[i] not in image_quality_for_each_person:
                            image_quality_for_each_person[identities[i]] = []
                        image_quality_for_each_person[identities[i]].append((quality, crop_img))

                        if len(image_quality_for_each_person[identities[i]]) > 10:
                            self.test = image_quality_for_each_person
                            image_quality_for_each_person[identities[i]] = list(sorted(image_quality_for_each_person[identities[i]], reverse=True, key=lambda x: x[0]))
                            image_quality_for_each_person[identities[i]].pop()

                    self.tracking_df = self.tracking_df.append({'frame': k, 'time': k * 1.0 / self.fps, 
                                                    'identity': identities, 'peds_bbx': bbx_xyxy, 
                                                    'head_bbx': bbx_head}, ignore_index=True)
                    
                    # draw boxes on pedestrian heads
                    # head_bbx_vis = draw_bboxes(head_bbx_vis, bbx_head, identities)
                    # img = draw_bboxes(img, bbox, identities)
                    peds_bbx_vis = draw_bboxes(peds_bbx_vis, bbx_xyxy, identities)
                
                # ======= visualize and save head bbx 
                # cv2.imwrite(os.path.join(head_bbx_dir, 'head_%s.png' % str(k)), head_bbx_vis)
                
                # ======= visualize and save peds bbx
                # cv2.imwrite(os.path.join(peds_bbx_dir, 'peds_%s.png' % str(k)), peds_bbx_vis)
                        
            # self.out_head.append_data(cv2.cvtColor(head_bbx_vis, cv2.COLOR_BGR2RGB))
            self.out_peds.append_data(cv2.cvtColor(peds_bbx_vis, cv2.COLOR_BGR2RGB))
            # self.out_uv.append_data(cv2.cvtColor(uv_vis, cv2.COLOR_BGR2RGB))
            # self.out_segm.append_data(cv2.cvtColor(segm_vis, cv2.COLOR_BGR2RGB))

            run_count += 1
        
            stdout.write("\rtotal frame count is %s/%s, %s people detected" % (str(k), str(self.frame_count), str(len(bbox))))
            stdout.flush()

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.person_df = self.reorder_df_object_wise(self.tracking_df)
        self.person_df['video_start'] = self.ts_start
        
        # ======= close video writing
        # self.out_segm.close()
        self.out_peds.close()
        # self.out_head.close()
        # self.out_uv.close()

        # ======= insert person data to database
        # self.person_df.to_sql('%s_%s_%s_track' % (self.client_id, self.floor_id, self.camera_id), self.engine, if_exists="append")

        # ======= write person image to files
        for p, persons in image_quality_for_each_person.items():
            frame_output_path = os.path.join(self.ped_img_dir, p)
            create_folder(frame_output_path)
            for l, (quality, person_image) in enumerate(persons):
                cv2.imwrite(os.path.join(frame_output_path, '%s.png' % str(l)), 
                            person_image)
        assert 0

        cv2.destroyAllWindows()

    def reorder_df_object_wise(self, tracking_df):
        person_df = pd.DataFrame(columns=['person_id', 'start_frame', 'end_frame', 'start_time', 
                                         'end_time', 'frames', 'trajectory', 'bbx', 'head_bbx'])
        # ['frame', 'time', 'identity', 'peds_bbx', 'head_bbx']
        # extend_df is extending frame-wise dataframe in to single frame single person per row
        extend_df = pd.DataFrame(columns=['frame', 'time', 'identity', 'peds_bbx', 'head_bbx'])
        for i, row in tracking_df.iterrows():
            for j, _ in enumerate(row['identity']):
                extend_df = extend_df.append({'frame': row['frame'], 'time': row['time'], 
                                                      'identity': row['identity'][j], 'peds_bbx': row['peds_bbx'][j], 
                                                      'trajectory': [int(row['peds_bbx'][j][2] - (row['peds_bbx'][j][2] - row['peds_bbx'][j][0]) * 0.15), 
                                                        int((row['peds_bbx'][j][3] + row['peds_bbx'][j][1]) / 2)],
                                                      'head_bbx': row['head_bbx'][j].tolist() if row['head_bbx'][j] is not None else [None, None, None, None]}, ignore_index=True)

        for person_id in extend_df['identity'].unique():
            filtered = extend_df[extend_df['identity'] == person_id]
            person_df = person_df.append({'person_id':person_id, 'start_frame':filtered['frame'].min(), 
                                        'end_frame':filtered['frame'].max(), 'start_time':filtered['time'].min(), 
                                        'end_time':filtered['time'].max(), 'frames':filtered['frame'].tolist(), 
                                        'bbx':filtered['peds_bbx'].tolist(), 'head_bbx': filtered['head_bbx'].tolist(), 
                                        'trajectory': filtered['trajectory'].tolist()},
                                        ignore_index=True)

        return person_df