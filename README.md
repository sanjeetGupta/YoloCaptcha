# YOLO CAPTCHA
### This is a YOLOv2 based Captcha security breaking model.


#### This work is an example of using synthetically generated data to train ML models.
#### Also, this is an indication of why Captcha Security is no longer useful and should be replaced with other technologies like reCAPTCHA

# Overview and Result
Accuracy for getting the captcha completely right! 
* Accuracy on  Generated Images: 89%
* Accuracy on Real Target Captcha Images: 71% (Basically, this model will break the captcha security in an average of 1.5-2 tries)


what does the Yolo captcha model do?

<img src="https://github.com/sanjeetGupta/YoloCaptcha/blob/master/images/example1_map.png" width="900" height="150" />


Some more Results (Inference done in example.ipynb) :

<img src="https://github.com/sanjeetGupta/YoloCaptcha/blob/master/images/example2.png" width="300" height="100" />

Result: RL6F83

<img src="https://github.com/sanjeetGupta/YoloCaptcha/blob/master/images/example4.png" width="300" height="100" />

Result: LP72Z


# Data Generation

It is not practical to download captchas and manually draw bounding boxes.
So, we have to automatically generate them. 

<i><b>generating_capcha.py</b></i> creates captcha using OpenCV and Pillow.

The main conclusions from my experiments with generating captcha to solve the captcha at a target website 
 *  The generated captcha should be similar to the captcha we are trying to break. (Obviously !)
 *  Noise Rocks!! It’s not enough if the generated captcha just looks similar, the pixel value distribution should also be similar. This can be achieved by adding random noise, blurring the image using median or gaussian blur.
 *  It's a balance between trying to achieve random enough generated images to generalize well and also to make sure the generated images don't have a very different pixel distribution than the target images.

Pixel distribution Real vs Generated  
<img src="https://github.com/sanjeetGupta/YoloCaptcha/blob/master/images/real_captcha_dist.png" width="300" height="300" />  <img src="https://github.com/sanjeetGupta/YoloCaptcha/blob/master/images/generated_captcha_dist.png" width="300" height="300" />


# Training Procedure 

Refer train_capcha.ipynb for complete code

* Step 1: Build the model architecture using Keras

         from create_model import build_yolo_model
         model = build_yolo_model() 
* Step 2: Download pre-trained YOLO weights and load the weights into the model (Required to convert the weights )

        ! wget https://pjreddie.com/media/files/yolov2.weights
        from utils import load_model_from_original_yolo_weights
        model = load_model_from_original_yolo_weights(model,'yolov2.weights')
        
        
* Step 3: Randomize the final layer weights.
        
        this step is performed by load_model_from_original_yolo_weights itself

* Step 4: Define Loss function
        
        from utils import custom_loss
            
* Step 5: Generate Captchas
        
        from generating_capcha import capcha_list
        from utils import BatchGenerator
        all_imgs = capcha_list()
        train_batch = BatchGenerator(images=all_imgs, config=generator_config,seed_start=0,seed_end=10000, norm=normalize)

* Step 6: Train on captcha Data
    ```
    model.fit_generator(generator = train_batch, steps_per_epoch  = len(train_batch), epochs = 100)
    ```
           
# Inference and Post Processing 

The YOLO Captcha Model outputs a lot of bounding boxes, we have to filter out extra boxes and get the right set of boxes.
This can be done by applying the following constraints :
* Object Probability. (ot)
* Non - Maximum Suppression. (nt)
* Number of boxes. (nb)
* Kinds of characters. (char_types)

Example: We can take only 6 boxes where the character is either upper-case or a digit with ot=0.3, nt=0.3.
 

Refer example.ipynb for complete code
 * Load Model 
    ```
    model=build_yolo_model('weights.h5')
    ```
 * Prediction and Post Processing
    ```
    solve_capcha.solve_capcha(capcha_path,model,capcha_length=5,char_types=['upper','digit'],show_box=True,ot=0.3,nt=0.2)

    ```
