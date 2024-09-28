from PIL import Image,ImageQt
import cv2
import glob
import os

import paddle
import paddle.vision.transforms as transforms
from paddle.vision.transforms import Resize,Normalize,Transpose
import numpy as np
from paddle.nn import functional as F
import interpretdl as it

from model.resnet_model import resnet50
# from model.mobilevit import mobilevit_s
from model.poolformer_attention import poolformer_m48
# from model.vip_model import vip_s14,vip_m7
# from model.vit import VisionTransformer
from model.ConvNeXt import convnext_base
# from model.swinT import swin_b

import gradio as gr 

device = paddle.device.get_device()
paddle.device.set_device(device) 
# model = resnet50(pretrained=False,num_classes = 2)

############################### Set parameters ####################################

networksDict = {
    'poolformer':poolformer_m48(num_classes = 2,use_attention=False,activations=False), 
    'convnext':convnext_base(num_classes=2,activations = False),
}

poolformerCheckpointDict = {
            'Baseline(10%)':"checkpoints/poolformer_baseline_0.1/poolformer_0.1_baseline.pdparams",
            'Upperbound(100%)':"checkpoints/poolformer_upperbound/params_best.pdparams",
            'SCDP(10%)':"checkpoints/poolformer_SRC_MT_0.1/params_best.pdparams"
        }

convnextCheckpointDict = {
            'SRC-MT(10%)':"checkpoints/convnext_SRC_MT_0.1/convnext_SRC_MT_0.1.pdparams",
            
        }

model_checkpoint_dict = {
    "poolformer:Baseline(10%)":[networksDict["poolformer"],poolformerCheckpointDict['Baseline(10%)']],
    "poolformer:Upperbound(100%)":[networksDict["poolformer"],poolformerCheckpointDict['Upperbound(100%)']],
    "poolformer:SCDP(10%)":[networksDict["poolformer"],poolformerCheckpointDict['SCDP(10%)']],
    "convnext:SRC-MT(10%)":[networksDict["convnext"],convnextCheckpointDict['SRC-MT(10%)']],
}

model_options = ["poolformer:Baseline(10%)","poolformer:Upperbound(100%)","poolformer:SCDP(10%)","convnext:SRC-MT(10%)"]


############################### define functions ####################################
# image preprocessing：

def transImg(img):
    testTransform = transforms.Compose([Resize(size=(224, 224)),Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'),transforms.ToTensor()])
    origin_img = img    
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img))
    img = img.convert('RGB')
    img = testTransform(img)
    img = paddle.reshape(img, shape=[1] + list(img.shape)) # <3,224,224> --> <1,3,224,224>
    # print(type(img))
    return origin_img,img

def predict(img,model_option):
    
    model = model_checkpoint_dict[model_option][0]
    checkpoint_path = model_checkpoint_dict[model_option][1]
    checkpoint = paddle.load(checkpoint_path)
    model.set_dict(checkpoint)    
    
    origin_img,input = transImg(img)                
    # print("Input Type:",type(input))
    with paddle.no_grad():
        output = model(input)
    outputSoftmax = F.softmax(output,axis=-1)
    outputArray = outputSoftmax.numpy()
    # print(outputArray)
    maxIndex = np.argmax(outputArray,axis=-1)
    if maxIndex == 0:
        output = "Unfortunately, you have been diagnosed with AS"
    elif maxIndex == 1:
        output = "Congratulations, you have been diagnosed as non-AS"
        
    # interpretation：
    input_array = np.array(input)
    print("inputArray Shape:",input_array.shape)

    modelGradCAMInterpreter = it.GradCAMInterpreter(model,device)
    model_result = [n for n,v in model.named_sublayers()]
    if 'network.4.20.mlp' in model_result:
        target_layer_name='network.4.20.mlp'
    else:
        target_layer_name=model_result[-10]
        
    heatmapArray = modelGradCAMInterpreter.interpret(
        input_array,
        target_layer_name=target_layer_name,
        # label=0,  
        visual=False  
    )
    print(type(heatmapArray),heatmapArray.shape)                
    # Get interpretable images
    if not isinstance(img,np.ndarray):
        img = img.convert('RGB') # <W,H,C>
    else:
        print("img_shape",img.shape)
    originHeight,originWidth = img.shape[0],img.shape[1]
    img = np.array(img)
    # img = cv2.resize(img,(224,224))
    if len(img.shape) == 4:
        img = img[0]
    if np.max(img) <= 1:   
        img *= 255.0
    assert len(img.shape) == 3,"for one image only"
    if img.shape[2] != 3:
        img = img.transpose((1,2,0))
    print("img shape:",img.shape)
    print("self.img size:",img.size)

    if len(heatmapArray.shape) == 3:
                assert heatmapArray.shape[0] == 1 , "For one image only"
                heatmapArray = heatmapArray[0]
                assert len(heatmapArray.shape) == 2
    heatmapArray = (heatmapArray - np.min(heatmapArray)) / (np.max(heatmapArray) - np.min(heatmapArray))
    heatmapArray = cv2.resize(heatmapArray,(224,224))
    heatmapArray = np.uint8(255*heatmapArray)
    heatmapArray = cv2.applyColorMap(heatmapArray, cv2.COLORMAP_JET)
    heatmapArray = cv2.cvtColor(heatmapArray, cv2.COLOR_BGR2RGB)
    
    heatmap = cv2.resize(heatmapArray,(originWidth,originHeight))        

    print("heatmap shape:",heatmap.shape)
    overlayVis = img*0.6 + heatmap*0.4
    overlayVis = cv2.resize(overlayVis,(originWidth,originHeight))
    
    overlayHeatmapImage = Image.fromarray(np.uint8(overlayVis))
    # overlayHeatmapImage = getOverlayHeatmap(img,heatmapArray)
    return overlayHeatmapImage,output

############################### Start ####################################

title = "A Novel Semi-supervised Learning Model Based On Pelvic Radiographs For Ankylosing Spondylitis Diagnosis Reduces 90% Of Annotation Cost"

demo = gr.Interface(
    inputs = [gr.Image(), 
              gr.inputs.Dropdown(choices=model_options, label="Please choose model"),
              ], # “image“==gr.Image()
    # inputs=[gr.Webcam()], 
    outputs=["image","text"],
    fn=predict,
    title = title,
    examples = [[str(img_path)] for img_path in glob.glob("example/*")],  # some examples
    )

demo.launch()


