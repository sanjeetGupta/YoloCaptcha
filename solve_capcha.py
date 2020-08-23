import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import WeightReader, decode_netout, draw_boxes,BoundBox
from constants import LABELS




def compress_df_dis(df):
    x_c=list(df['x'])
    x_dis=list(np.array(x_c[1:])-np.array(x_c[0:-1]))
    x_dis.append(999)
    df['x_dis']=x_dis
    arg=df.x_dis.argmin()
    drop_arg=df.iloc[arg:arg+2,:]['score'].argmin()
    df.drop(drop_arg,inplace=True)
    df=df.reset_index(drop=True)
    return df


def compress_df_xmin_dis(df):
    x_c=list(df['xmin'])
    x_dis=list(np.array(x_c[1:])-np.array(x_c[0:-1]))
    x_dis.append(999)
    df['x_dis']=x_dis
    arg=df.x_dis.argmin()
    drop_arg=df.iloc[arg:arg+2,:]['score'].argmin()
    df.drop(drop_arg,inplace=True)
    df=df.reset_index(drop=True)
    return df

def compress_df_iou(df):
    ious=[]
    for i in range(df.shape[0]-1):
        box1=df.iloc[i]
        box2=df.iloc[i+1]
        intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        iou=float(intersect) / union
        ious.append(iou)
    ious.append(-999)
    df['iou']=ious
    arg=df.iou.argmax()
    drop_arg=df.iloc[arg:arg+2,:]['score'].argmin()
    df.drop(drop_arg,inplace=True)
    df=df.reset_index(drop=True)
    return df


def char_type(char):
    if char.isupper():
        return 'upper'
    if char.islower():
        return 'lower'
    if char.isdigit():
        return 'digit'
    
def solve_capcha(capcha_path,model,capcha_length=5,path=True,char_types=['upper','lower','digit'],ot=0.05,nt=0.05,compress_function='dis',show_box=False):
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    if path==True:
        capcha = cv2.imread(capcha_path)
    else:
        capcha=capcha_path
    plt.imshow(capcha[:,:,::-1])
    if show_box: image_box=capcha
    capcha = cv2.resize(capcha, (416, 416))
    capcha = capcha / 255.
    capcha = capcha[:,:,::-1]
    capcha = np.expand_dims(capcha, 0)
    netout = model.predict([capcha, dummy_array])
    boxes = decode_netout(netout[0], obj_threshold=ot,nms_threshold=nt,anchors=ANCHORS, nb_class=CLASS)
    if show_box:
        image_box= draw_boxes(image_box, boxes, labels=LABELS)
        plt.imshow(image_box[:,:,::-1]); plt.show()
    xmin=[]
    xmax=[]
    ymin=[]
    ymax=[]
    chars=[]
    score=[]
    for b in boxes:
        xmin.append(b.xmin)
        xmax.append(b.xmax)
        ymin.append(b.ymin)
        ymax.append(b.ymax)
        chars.append(LABELS[b.get_label()])
        score.append(b.get_score())
    df=pd.DataFrame()
    df['char']=chars
    df['score']=score
    df['xmin']=xmin
    df['xmax']=xmax
    df['ymin']=xmin
    df['ymax']=xmax    
    df['x']=(df.xmin+df.xmax)/2
    df['y']=(df.ymin+df.ymax)/2
    df['char_type']=df.char.apply(lambda x: char_type(x))
    df=df[df.char_type.apply(lambda x: x in char_types)]
    df_r=df.copy()
    if compress_function=='dis':
        compress_df_func=compress_df_dis
        df=df.sort_values('x')
    if compress_function=='xmin_dis':
        compress_df_func=compress_df_xmin_dis
        df=df.sort_values('xmin')
    if compress_function=='iou': 
        compress_df_func=compress_df_iou
        df=df.sort_values('x')
    df=df.reset_index(drop=True)
    num_rows=df.shape[0]
    while num_rows > capcha_length:
        df=compress_df_func(df)
        num_rows=df.shape[0]
    return df_r,df.char.str.cat()

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

        
        
