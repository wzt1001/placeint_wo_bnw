from parser_hopson_one import *
import datetime
import time

import json
from helpers.db_opt import connect_to_db
from sqlalchemy import create_engine
import boto3
import os
import pandas as pd
from helpers.s3_module import *
import subprocess
import yaml
from io import StringIO

config_path = 'config.json'
client_id = 'beijing_hopson_one'
# video_name = '2019-12-22-100000_L2-J11_hopson.mp4'

# list all buckets on s3
s3 = boto3.resource('s3')
s3_file_sys = FileSysModule()
bucket_name = 'beijing-hopson-one-hk-new'
bucket = boto3.resource('s3').Bucket(bucket_name)
prefix = 'uploads/hopson_one_beijing_II'

with open(config_path) as jsonfile:
    config = json.load(jsonfile)
    
engine = create_engine('''postgresql://%s:%s@%s:%s/%s''' % (config['db']['user'], 
                                              "placeint", config['db']['host'], str(5432),
                                              "placeint"))
con, cur = connect_to_db(user=config['db']['user'], pwd="placeint", 
                host=config['db']['host'], 
                port=5432, db="placeint")

def main(gpu)

    for object in bucket.objects.all():

        if object.key[:29] == prefix and object.key.split('.')[-1] == 'mp4':

            basename = os.path.basename(object.key)
            ts_start, file_size, byte_per_sec, duration, camera_id, floor_id = parse_hopson_one(basename)
            down_path = os.path.join('../data/input', basename)

            # ==========================
            # =   re-write yaml file   =
            # ==========================

            fname = "experiments/cfgs/tracktor.yaml"
            stream = open(fname, 'r')
            data = yaml.load(stream)
            data['tracktor']['video_filename'] = basename
            data['tracktor']['gpu'] = str(gpu)
            with open(fname, 'w') as yaml_file:
                yaml_file.write(yaml.dump(data, default_flow_style=False))
                
            # ==========================
            # =   pass on unselected cameras
            # ==========================

            # if not (camera_id == 'J08' or camera_id == 'J05' or camera_id == 'J16' or camera_id == 'J15'):
            if camera_id not in ['J16', 'J34', 'J35', 'J33', 'J30', 'J29', 'J28', 'J26', 
                                'J22', 'J19', 'J15', 'J14', 'J13', 'J11', 'J10', 
                                'J08', 'J06']:
                print('--- skipping %s' % basename)
                continue
            else:
                print('--- processing %s' % basename)

            # ==========================
            # =   check if db has row
            # ==========================

            query = '''SELECT COUNT(*) FROM public.processes WHERE filename = '%s' ''' % (basename)
            cur.execute(query)
            con.commit()

            if cur.fetchone()[0] == 0:
                print('adding row to processing table')
                insert_record = pd.DataFrame({'filename': basename, 'client_id': client_id, 'ts_start': ts_start, 
                                'duration': int(duration), 'step_0_detecting_and_tracking': 0, 
                                'step_1_gaze_estimation': 0, 'step_2_classification': 0,
                                'step_3_kpi': 0}, index=[0])
                insert_record.to_sql('processes', engine, if_exists="append", index=False)
            else:
                print('row already exist in processing table')


            if not os.path.exists(down_path):
                print('--- downloading', basename)
                s3_file_sys.downloadfile("uploads/hopson_one_beijing_II/" + basename, down_path)
                print('--- finished downloading', basename)

            # ==========================
            # =   check if already ran
            # ==========================

            query = '''SELECT COUNT(*) FROM public.processes WHERE filename = '%s' AND %s > 0''' % (basename, 'step_0_detecting_and_tracking')
            cur.execute(query)
            con.commit()
            if cur.fetchone()[0] >= 1:
                continue

            # ==========================
            # =   start subprocess
            # ==========================
            
            # set state to 10 to prevent other processes from running
            query = '''UPDATE public.processes SET %s = 10 WHERE filename = '%s' ''' % ('step_0_detecting_and_tracking', basename)
            cur.execute(query)
            con.commit()

            cmd = ['python','test_tracktor.py']
            p = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)

            process_output, _ =  p.communicate()

            # def log_subprocess_output(pipe):
            #     for line in iter(pipe.readline, b''): # b'\n'-separated lines
            #         print('got line from subprocess: %r', line)
            # log_subprocess_output(process_output)

            # try:
            #     # Filter stdout
            #     for line in iter(p.stdout.readline, ''):
            #         sys.stdout.flush()
            #         # Print status
            #         print(">>> " + line.rstrip())
            #         sys.stdout.flush()
            # except:
            #     sys.stdout.flush()

            # Wait until process terminates (without using p.wait())
            while p.poll() is None:
                # Process hasn't exited yet, let's wait some time
                time.sleep(0.5)

            # Get return code from process
            return_code = p.returncode

            if bool(return_code) == False:
                assert 0

            else:
                print("--- completed")

            query = '''UPDATE public.processes SET %s = 30 WHERE filename = '%s' ''' % ('step_0_detecting_and_tracking', basename)
            cur.execute(query)
            con.commit()

    con.close()
    
if __name__ == '__main__':
    gpu = sys.argv[1]
    main()
    
