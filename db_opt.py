import pandas as pd
import psycopg2
# from sqlalchemy import create_engine
import math

# def psql_insert_copy(table, conn, keys, data_iter):
#     # gets a DBAPI connection that can provide a cursor
#     dbapi_conn = conn.connection
#     with dbapi_conn.cursor() as cur:
#         s_buf = StringIO()
#         writer = csv.writer(s_buf)
#         writer.writerows(data_iter)
#         s_buf.seek(0)

#         columns = ', '.join('"{}"'.format(k) for k in keys)
#         if table.schema:
#             table_name = '{}.{}'.format(table.schema, table.name)
#         else:
#             table_name = table.name

#         sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
#             table_name, columns)
#         cur.copy_expert(sql=sql, file=s_buf)

# engine = create_engine('postgresql://myusername:mypassword@myhost:5432/mydatabase')
# track_df.to_sql('table_name', engine, method=psql_insert_copy)


# connect to database
def connect_to_db(user="postgres", pwd="", host="127.0.0.1", port=5432, db="placeint"):
    try:
        con = psycopg2.connect(user = user,
                                  password = pwd,
                                  host = host,
                                  port = port,
                                  database = db)
        cur = con.cursor()
        # Print PostgreSQL Connection properties
        print ( con.get_dsn_parameters(),"\n")
        # Print PostgreSQL version
        cur.execute("SELECT version();")
        record = cur.fetchone()
        print("You are connected to - " + str(record) + "\n")
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL" + str(error))
    return con, cur

# insert information on each video to video_info table
def insert_video_info(cur, con, client_id, zone_id, bld_id, cam_id, ts_start, ts_end, duration, fps, created_at, video_id, table):

    try:
        query = '''insert into public.%s (client_id, zone_id, bld_id, 
                    cam_id, ts_start, ts_end, duration, fps, created_at, video_id) values 
                    ('%s', '%s', '%s', '%s', '%s', '%s', %s, %s, '%s', '%s')
                ''' % (table, client_id, zone_id, bld_id, cam_id, ts_start, ts_end, 
                       duration, fps, created_at, video_id)
        print(query)
        cur.execute(query)
        con.commit()
    except Exception as e:
        print('inserting video information failed', e)
    return

def create_insert_db(cur, con, track_df, attr_df, video_id):
    
    # create table for bounding box
    query = '''drop table if exists public.%s_bbx; create table if not exists public.%s_bbx (
            person_id text,
            frame integer,
            time real,
            x_0 real,
            y_0 real,
            x_1 real,
            y_1 real
        );''' % (video_id, video_id)
    cur.execute(query)
    print('Created table public.%s_bbx' % video_id)

    # insert values for bounding box
    for index, row in track_df.iterrows():
        query = '''
            insert into public.%s_bbx values (
            '%s', %s, %s, %s, %s, %s, %s
            )
        ''' % (video_id, row['person_id'], row['frame'], row['time'], 
               row['x_0'], row['y_0'], row['x_1'], row['y_1'])
        cur.execute(query)
    con.commit()
    print('Finished inserting into table public.%s_bbx' % video_id)

    # create table for trajectory
    query = '''drop table if exists public.%s_traj; create table if not exists public.%s_traj (
            person_id text,
            start_frame integer,
            end_frame integer,
            start_time real,
            end_time real,
            framestamp integer[],
            timestamp real[],
            path geometry,
            bbx real[][],
            velocity real[]
        );''' % (video_id, video_id)
    cur.execute(query)
    con.commit()
    print('Created table public.%s_traj' % video_id)

    # insert values for trajectory
    # path '[(0,0),(1,1),(2,0)]'

    for person_id in track_df['person_id'].unique():
        min_frame = track_df[track_df['person_id'] == person_id]['frame'].min()
        max_frame = track_df[track_df['person_id'] == person_id]['frame'].max()
        min_time  = track_df[track_df['person_id'] == person_id]['time'].min()
        max_time  = track_df[track_df['person_id'] == person_id]['time'].max()
        framestamp= str(track_df[track_df['person_id'] == person_id]['frame'].values.tolist()).replace("'",'').replace('[','{').replace(']', '}')
        timestamp = str(track_df[track_df['person_id'] == person_id]['time'].values.tolist()).replace("'",'').replace('[','{').replace(']', '}')
        path      = track_df[track_df['person_id'] == person_id][['anchor_x', 'anchor_y']].values.tolist()
        path      = [str(a[0]) + ' ' + str(a[1]) for a in path]
        path      = str(path).replace("'",'').replace('[','').replace(']', '')
        bbx       = str(track_df[track_df['person_id'] == person_id][['x_0', 'x_1', 'y_0', 'y_1']].values.tolist()).replace("'",'').replace('[','{').replace(']', '}')
        velocity  = track_df[track_df['person_id'] == person_id][['center_x', 'center_y', 'time']].values.tolist()
        velocity  = [ math.sqrt((velocity[idx - 1][0] - v[0])**2 + 
                                (velocity[idx - 1][1] - v[1])**2) / (v[2] - velocity[idx - 1][2]) for
                                 idx, v in enumerate(velocity)]
        # first point adjusted to be the same value as the second
        velocity[0] = velocity[1]

        # stringify so that it can be inserted into database
        velocity  = str(velocity).replace('[', '{').replace(']', '}')

        # if path is too short, ignore it 
        if len(path) < 5:
            continue

        query = '''
            insert into public.%s_traj values (
            '%s', %s, %s, %s, %s, '%s', '%s', ST_GeomFromText('LINESTRING(%s)'), '%s', '%s'
            )
        ''' % (video_id, person_id, min_frame, max_frame, min_time, max_time, 
               framestamp, timestamp, str(path), bbx, velocity)
        cur.execute(query)
    con.commit()
    print('Finished inserting into table public.%s_traj' % video_id)

    # create table for human attributes
    query = '''drop table if exists public.%s_attr; create table if not exists public.%s_attr (
            person_id text,  personalLess30 integer, personalLess45 integer, personalLess60 integer, 
            personalLarger60 integer, carryingBackpack integer, carryingOther integer, lowerBodyCasual integer, 
            upperBodyCasual integer, lowerBodyFormal integer, upperBodyFormal integer, accessoryHat integer, 
            upperBodyJacket integer, lowerBodyJeans integer, footwearLeatherShoes integer, upperBodyLogo integer, 
            hairLong integer, personalMale integer, carryingMessengerBag integer, accessoryMuffler integer, 
            accessoryNothing integer, carryingNothing integer, upperBodyPlaid integer, carryingPlasticBags integer, 
            footwearSandals integer, footwearShoes integer, lowerBodyShorts integer, upperBodyShortSleeve integer, 
            lowerBodyShortSkirt integer, footwearSneaker integer, upperBodyThinStripes integer, 
            accessorySunglasses integer, lowerBodyTrousers integer, upperBodyTshirt integer, upperBodyOther integer, 
            upperBodyVNeck integer
        );''' % (video_id, video_id)
    cur.execute(query)
    print('Created table public.%s_attr' % video_id)

    for index, row in attr_df.iterrows():
        query = '''
            insert into public.%s_attr values (
            '%s', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )''' % (video_id, row.name, row['personalLess30'], row['personalLess45'], row['personalLess60'], 
               row['personalLarger60'], row['carryingBackpack'], row['carryingOther'], row['lowerBodyCasual'], 
               row['upperBodyCasual'], row['lowerBodyFormal'], row['upperBodyFormal'], row['accessoryHat'], 
               row['upperBodyJacket'], row['lowerBodyJeans'], row['footwearLeatherShoes'], row['upperBodyLogo'], 
               row['hairLong'], row['personalMale'], row['carryingMessengerBag'], row['accessoryMuffler'], 
               row['accessoryNothing'], row['carryingNothing'], row['upperBodyPlaid'], row['carryingPlasticBags'], 
               row['footwearSandals'], row['footwearShoes'], row['lowerBodyShorts'], row['upperBodyShortSleeve'], 
               row['lowerBodyShortSkirt'], row['footwearSneaker'], row['upperBodyThinStripes'], 
                    row['accessorySunglasses'], row['lowerBodyTrousers'], row['upperBodyTshirt'], 
                    row['upperBodyOther'],row['upperBodyVNeck'])
        cur.execute(query)
    con.commit()
    print('Finished inserting into table public.%s_attr' % video_id)
    
    print('Database insertion is complete')

def get_traj_join_attr(cur, con, video_id):
    query = '''select a.person_id, a.start_frame, a.end_frame, a.start_time, a.end_time, a.framestamp, a.timestamp,
            a.path, a.bbx, a.velocity, b.personalLess30, b.personalLess45, b.personalLess60,
            b.personalLarger60, b.personalMale
            from public.%s_traj a join public.%s_attr b on a.person_id = b.person_id ''' % (video_id, video_id)
    cur.execute(query)
    con.commit()

    results = cur.fetchall()
    return results
