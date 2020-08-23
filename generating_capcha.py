import cv2
import numpy as np
import random,string
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


class capcha_list():
    """
    A generator class to create captcha image given a random seed.
    """
    def __init__(self):
        return
    def __getitem__(self, seed,type=1):
        if type ==1:
            return capcha_gen_annotate(seed,img_h=30,img_w=100,text_length=6)
        else:
            return italic_capcha(seed, img_h=30, img_w=100, text_length=6)




def array2PIL(arr):
    """
    convert np array to PIL image
    """
    size=(arr.shape[1],arr.shape[0])
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def PIL2array(img):
    """
    Convert PIL image to np array
    """
    return np.array(np.array(img,np.uint8)[:,:,:3])



def capcha_gen_annotate(seed,img_h=60,img_w=200,text_length=5):
    """
    Generates a captcha with a random seed
    """
    img_dict={}
    img_dict['height']=img_h
    img_dict['width']=img_w
    obj_list=[]
    img=np.ones((img_h,img_w,3))
    img[:]=(254,254,254)
    random.seed(seed)
    fonts=[cv2.FONT_HERSHEY_DUPLEX,cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_SIMPLEX,cv2.FONT_HERSHEY_TRIPLEX,cv2.FONT_ITALIC]
    lineTypes=[1,2,3]
    a=random.choice(list(range(0,254)))
    b=random.choice(list(range(0,254)))
    c=0
    color_list=[a,b,c]
    random.shuffle(color_list)
    font                   = random.choice(fonts)
    fontScale              = random.randint(10,16)/10
    fontColor              =tuple(color_list)
    lineType               =random.choice(lineTypes)
    ts                     =[10,12,15,17,20,22,25]
    ys                     =[50,52,45,47,40,42,35,37,30]
    t                      =random.choice(ts)
    spas                   =[25,27,30,32,35]
    spa                    =random.choice(spas)
    capcha_text            =''
    thick                  =random.choice([2,3,4])

    for i in range(text_length):
        obj_dict={}
        font= random.choice(fonts)
        p=t+i*spa
        y=random.choice(ys)
        char=random.choice(list(string.ascii_uppercase+string.digits))
        img=cv2.putText(img=img,text=char, org=(p,y),fontFace=font,fontScale=fontScale,color =fontColor,lineType=lineType,thickness=thick)
        textSize = cv2.getTextSize(text=char, fontFace=font, fontScale=fontScale,thickness=thick)
        obj_dict['name']=char
        obj_dict['xmin']=p
        obj_dict['ymin']=y-textSize[0][1]
        obj_dict['xmax']=p+textSize[0][0]
        obj_dict['ymax']=y
        obj_list.append(obj_dict)
        capcha_text=capcha_text+char
    img_dict['object']=obj_list
    density_noise=random.choice([100,200,150,50,300])
    for i in range(density_noise):
        x=random.choice(range(0,img_h))
        y=random.choice(range(0,img_w))

        bw=random.choice([2,1,3,3,2,2,1,1,4])
        bh=random.choice([2,1,3,3,2,2,1,1,4])
        x2=np.min([x+bh,img_h])
        y2=np.min([y+bh,img_w])
        darkness=random.choice([10,15,20,30,40,50])
        img[x:x2,y:y2,:]=np.zeros((x2-x,y2-y,3))+darkness

    noise_frac=random.choice([5,10,15])
    img=np.random.normal(loc=img,scale=(img/noise_frac)+1)
    img=np.clip(img, 0, 255)
    img=img.astype(np.uint8)
    probability=random.random()
    if probability > 0.5:
        img=cv2.GaussianBlur(img,(3,3),0)
    if 0.2< probability <=0.5:
        img=cv2.medianBlur(img,3)
    img_dict['filename']=img
    return img_dict




def italic_capcha(seed,img_h=30,img_w=100,text_length=6):
    """
    Generate a italic char captcha with a random seed
    """
    random.seed(seed)
    img_dict={}
    img_dict['height']=img_h
    img_dict['width']=img_w
    obj_list=[]
    img=np.ones((img_h,img_w,3))
    img[:]=(254,254,254)
    img=img.astype('uint8')
    back=img
    a=random.choice(list(range(0,254)))
    b=random.choice(list(range(0,254)))
    c=0
    color_list=[a,b,c]
    random.shuffle(color_list)
    fontColor =tuple(color_list)
    t =random.choice(list(range(8,12)))
    spa =random.choice(list(range(10,16)))
    y=random.choice(list(range(8,12)))
    capcha_text= ''
    italic_ttfs=['ttf_fonts/Raleway-Italic.ttf',
                 "ttf_fonts/BOD_I.TTF",
                'ttf_fonts/timesi.ttf',
                'ttf_fonts/timesbi.ttf',
                'ttf_fonts/SourceSansPro-SemiBoldItalic.ttf',
                'ttf_fonts/cambriai.ttf',
                'ttf_fonts/BOD_I.TTF',
                'GothamLightItalic.ttf']
    fsize=random.choice(list(range(14,20)))
    font = ImageFont.truetype(random.choice(italic_ttfs), fsize)
    for i in range(text_length):
        obj_dict={}
        p=t+i*spa
        char=random.choice(list(string.ascii_uppercase+string.digits+string.ascii_lowercase))
        img=array2PIL(img)
        draw = ImageDraw.Draw(img)

        draw.text((p, y),char,fontColor,font=font)
        textSize=draw.textsize(char, font=font)
        img=PIL2array(img)
        obj_dict['name']=char
        obj_dict['xmin']=p
        obj_dict['ymin']=y
        obj_dict['xmax']=p+textSize[0]
        obj_dict['ymax']=y+textSize[1]
        obj_list.append(obj_dict)
        capcha_text=capcha_text+char
    img_dict['object']=obj_list
    noise_frac=random.choice([15,16,17,18])
    img=np.random.normal(loc=img,scale=(img/noise_frac)+1)
    img=np.clip(img, 0, 255)
    img=img.astype(np.uint8)
    probability=random.random()
    if probability>0.5:
        img=cv2.GaussianBlur(img,(1,1),0)
    if 0.2<probability<=0.5:
        img=cv2.medianBlur(img,1)
    img_dict['filename']=img
    return img_dict


