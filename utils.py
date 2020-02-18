import psycopg2
import os, sys
import numpy as np
import cv2
import random
from scipy.ndimage import gaussian_filter1d


# generate random colors
def random_color():    
    rgb=(int(random.uniform(0.5, 1) * 255), 
          int(random.uniform(0.5, 1) * 255),
          int(random.uniform(0.5, 1) * 255))
    return rgb

# perform perspective transformation
def four_point_transform(src, new_h, new_w):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = src
    
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    
    # redefine width and height
    maxHeight = new_h
    maxWidth  = new_w
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(src, dst)
    
    # return the warped image
    return M

# smoothing lines
def smooth_line(a):
    x, y, time = a.T
    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100)

    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    sigma = 5
    x3 = gaussian_filter1d(x2, sigma)
    y3 = gaussian_filter1d(y2, sigma)

    x4 = np.interp(t, t2, x3)
    y4 = np.interp(t, t2, y3)
    
    return np.float32([x4, y4, time])

# render matplot table on each frame of the output video
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

# transform matplot figure to numpy array
def fig_to_np(fig):
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis('off')
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))
    return X

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# transform mask to bbox
def mask_to_bbox(mask):
    #Code from Detectron modifyied to make the box 30% bigger
    """Compute the tight bounding box of a binary mask."""

    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]
    # print(mask.shape)
    W = mask.shape[0]
    H = mask.shape[1]
    
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]

    y0 = ys[0]
    y1 = ys[-1]
    w = x1-x0
    h = y1-y0
    
    x0 = max(0,x0-w*0.15)
    x1 = max(0,x1+w*0.15)
    y0 = max(0,y0-h*0.15)
    y1 = max(0,y1+h*0.15)

    return np.array((x0, y0, x1, y1), dtype=np.float32)

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

# draw bboxes on image
def draw_bboxes(img, bbox, identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        if box is None:
            continue
        x1,y1,x2,y2 = [int(i) for i in box]
        # assert 0, str([x1,y1,x2,y2])
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        # if is string, round it first 
        
        if isinstance(identities[i], str):
            id = (int(identities[i], 16)) % len(COLORS_10) if identities is not None else 0  
        elif isinstance(identities[i], int):
            id = (identities[i] % len(COLORS_10)) if identities is not None else 0  
        else:
            assert 0, "the ID type is unsupported for drawing boxes!"
        color = COLORS_10[id]
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        # assert 0, str([(x1, y1),(x2,y2),color,3])
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
    return img

# transform x0y0 to xcenter ycenter w h
def _xyxy_to_xcycwh(bbox_xyxy):
    x1,y1,x2,y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    return xc,yc,w,h

# 