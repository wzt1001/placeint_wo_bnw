import os
import time
from os import path as osp
import json
import logging
import sys
import cv2
from datetime import timedelta
import pytz
from sqlalchemy import create_engine
sys.path.append('~/data/placeint-main/scripts/')
from db_opt import connect_to_db
from utils import *
# from s3_module import *
from parser_hopson_one import  *

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='latest.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)
logger.handlers[0].stream = sys.stdout

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import imageio
import json
import hashlib
import pandas as pd

import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, get_mot_accum, evaluate_mot_accums
from video_loader import GeneralVideoDataset

ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


def reorder_df_object_wise(tracking_df):
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


@ex.automain
def main(tracktor, reid, _config, _log, _run):
    
    gpu = tracktor
    torch.cuda.set_device(int(tracktor['gpu']))

    video_filename = tracktor['video_filename']
    ts_start, file_size, byte_per_sec, duration, camera_id, floor_id = parse_hopson_one(tracktor['video_filename'])
    config_path = 'config.json'
    mark_data_path = 'mark.json'
    client_id = 'hopson_one'

    tracking_df = pd.DataFrame(columns=['frame', 'time', 'identity', 'peds_bbx', 'head_bbx'])
    image_quality_for_each_person = {}

    with open(config_path) as jsonfile:
        config = json.load(jsonfile)

    with open(mark_data_path) as jsonfile:
        mark_data = json.load(jsonfile)

    # basic information on the video
    video_path     = os.path.join('..', 'data', 'input', video_filename)
    video_id       = video_filename.split('.')[0]
    detect_per_sec = config['general']['detect_per_sec']
    PROCESS_GAZE_ESTIMATION = config['general']['PROCESS_GAZE_ESTIMATION']
    batch_size = 100

    # see if video exists
    if not os.path.exists(video_path):
        logger.error("video not found")
        assert 0, "video not found"

    # get cv2 video capture
    cap = cv2.VideoCapture('../data/input/%s' % video_filename)
    _, sample = cap.read()
    w = sample.shape[1]
    h = sample.shape[0]
    # cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
    engine = create_engine('''postgresql://%s:%s@%s:%s/%s''' % (config['db']['user'], 
                                                          "placeint", config['db']['host'], str(5432),
                                                          "placeint"))

    frame_count = int(cv2.VideoCapture.get(cap, int(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("fps %s, frame count %s, duration %s" % (str(fps), str(frame_count), str(duration)))

    frames_per_detect = int(int(fps) / int(detect_per_sec))

    # start end time of the video
    ts_end   = ts_start + timedelta(0, duration)
    logger.info("time start %s" % str(ts_start))
    logger.info("time end %s" % str(ts_end))

    # current time
    now = pytz.utc.localize(datetime.datetime.now())
    created_at = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info("created at %s" % str(created_at))

    # =================== to-do add export to video code and corresponding config keys
    # out_peds = out_peds = imageio.get_writer('./test_peds.mp4')
    ped_img_dir = os.path.join('./pedestrians_images', video_id)
    con, cur = connect_to_db(user=config['db']['user'], pwd="placeint", 
                    host=config['db']['host'], 
                    port=5432, db="placeint")

    ##########################
    #    Original function   #
    ##########################
    
    sacred.commands.print_config(_run)

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
    logging.info("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    mot_accums = []
    video_loader = GeneralVideoDataset(video_path, channels=3, batch_size=batch_size,
                                        fps=fps, frames_per_detect=frames_per_detect, effective_zone=mark_data[camera_id + '.png']["effective"])

    # dataset = Datasets(tracktor['dataset'])
    # for seq in dataset:
    tracker.reset()

    # k denote the count for a detect, k * frames_per_detect gives the real frame index
    k = 0
    while True:
        seq, valid = video_loader.get_new_batch()
        if valid == False:
            break
        # tracker.reset()
        print("processing frist batch")
        
        start = time.time()

        # logging.info(f"Tracking: {len(seq)}")
        data_loader = DataLoader(TensorDataset(seq), batch_size=1, shuffle=False)
        for i, frame in enumerate(tqdm(data_loader)):
            tracker.step({'img': frame[0].type(torch.cuda.FloatTensor).cpu()})
            # print(results)
            k += 1

        results = tracker.get_results()
        if bool(results):
            for i, frame in enumerate(data_loader):
                frame_idx = i + k - batch_size
                # hash the identities
                identities = [int(id) for id, traj in results.items() if frame_idx in traj]
                identities = [(hashlib.md5((str(j) + str(ts_start) + 
                            camera_id).encode('utf-8'))).hexdigest() for j in identities]
                bbx_xyxy = [[int(a) for a in traj[frame_idx][0:4]] for id, traj in results.items() if frame_idx in traj]
                bbx_head = [None for id, traj in results.items()]
                peds_bbx_vis = ((frame[0]).data.cpu().numpy() * 255)[0].transpose(1, 2, 0).astype(np.uint8)

                # ========================================
                # =     for exporting to video           =
                # ========================================
                # peds_bbx_vis = np.ascontiguousarray(peds_bbx_vis)
                # a = peds_bbx_vis.copy()
                # peds_bbx_vis = draw_bboxes(peds_bbx_vis, bbx_xyxy, identities)
                # out_peds.append_data(cv2.cvtColor(peds_bbx_vis, cv2.COLOR_BGR2RGB))


                # ========================================
                # =  get ped image quality, decide whether to save 
                # ========================================
                
                # calculate quality score by cropped image size and ratio, the best h/w ratio is 1.7 for now
                ratio = [(j[3] - j[1]) / (j[2] - j[0]) for j in bbx_xyxy]
                size = [(j[3] - j[1]) * (j[2] - j[0]) for j in bbx_xyxy]
                quality = [size[j] / (abs(ratio[j] - 1.7) + 0.5) for j in range(len(bbx_xyxy))]

                for j, e in enumerate(quality):

                    crop_img = peds_bbx_vis[bbx_xyxy[j][1]:bbx_xyxy[j][3], bbx_xyxy[j][0]:bbx_xyxy[j][2]]
                    print(identities[j])
                    if identities[j] not in image_quality_for_each_person:
                        image_quality_for_each_person[identities[j]] = []
                    image_quality_for_each_person[identities[j]].append((e, crop_img))

                    if len(image_quality_for_each_person[identities[j]]) > 10:
                        image_quality_for_each_person[identities[j]] = list(sorted(image_quality_for_each_person[identities[j]], 
                                                                       reverse=True, key=lambda x: x[0]))
                        image_quality_for_each_person[identities[j]].pop()

                tracking_df = tracking_df.append({'frame': frame_idx * frames_per_detect, 'time': frame_idx * frames_per_detect * 1.0 / fps, 
                                'identity': identities, 'peds_bbx': bbx_xyxy, 
                                'head_bbx': bbx_head}, ignore_index=True)


        time_total += time.time() - start

        logging.info(f"Tracks found: {len(results)}")
        logging.info(f"Runtime for {len(seq)}: {time.time() - start :.1f} s.")
        if tracktor['interpolate']:
            results = interpolate(results)
        # print(results)
        # mot_accums.append(get_mot_accum(results, seq))

        logging.info(f"Writing predictions to: {output_dir}")
        # seq.write_results(results, output_dir)

    # ======= write person image to files
    for p, persons in image_quality_for_each_person.items():
        frame_output_path = os.path.join(ped_img_dir, p)
        create_folder(frame_output_path)
        for l, (quality, person_image) in enumerate(persons):
            cv2.imwrite(os.path.join(frame_output_path, '%s.png' % str(l)), 
                        person_image)

    # ========================================
    # =  person_df reorder and put into db   =  
    # ========================================

    person_df = reorder_df_object_wise(tracking_df)
    person_df['video_start'] = ts_start
    person_df.to_sql('%s_%s_%s_track' % (client_id, floor_id, camera_id), engine, if_exists="append")

    # out_peds.close()q

