from PySide6.QtWidgets import QApplication,QWidget,QMainWindow,QFileDialog,QLabel,QPushButton
from Ui_AS_predict_app import Ui_MainWindow
from PIL import Image,ImageQt
import cv2

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

class MyWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)        
        self.bind()
    
    def bind(self):        
        # set device
        self.device = paddle.device.get_device()
        # initial
        self.lbShowPredict.setText("Welcome to the AS prediction system")   
        # get img
        self.pbSelectImg.clicked.connect(self.getImg) 
        # select model
        # get architecture
        self.networksDict = {
            'resnet50':resnet50(pretrained=False,num_classes=2,activations=False),
            'resnet50_CBAM':resnet50(pretrained=False,num_classes=2,use_CBAM_Module=True,activations=False),
            # 'vit':VisionTransformer(num_classes=2,), 
            'poolformer':poolformer_m48(num_classes = 2,use_attention=False,activations=False), # 一定要把activations设置成False
            # 'vip':vip_m7(num_classes = 2,),
            'poolformer_attention':poolformer_m48(num_classes = 2,use_attention=True,activations=False),
            'convnext':convnext_base(num_classes=2,activations = False),
            # 'swinT':swin_b(class_dim=2,),
        } 
        # get model weight
        self.poolformerCheckpointDict = {
            'Baseline(10%)':"checkpoints/poolformer_baseline_0.1/poolformer_0.1_baseline.pdparams",
            'Upperbound(100%)':"checkpoints/poolformer_upperbound/params_best.pdparams",
            'SCDP(10%)':"checkpoints/poolformer_SRC_MT_0.1/params_best.pdparams"
        }
        self.convnextCheckpointDict = {
            'SRC-MT(10%)':"checkpoints/convnext_SRC_MT_0.1/convnext_SRC_MT_0.1.pdparams",
            
        }
        
        # dict for model and weight
        self.modelCheckpointDict = {
            'poolformer':self.poolformerCheckpointDict,
            'poolformer_attention':{"None":"TODO"},
            'resnet50':{"None":"TODO"},
            'resnet50_CBAM':{"None":"TODO"},
            'convnext':self.convnextCheckpointDict,
        }
        
        # Aug
        self.testTransform = transforms.Compose([Resize(size=(224, 224)), Normalize(mean=[127.5, 127.5, 127.5],std=[127.5, 127.5, 127.5],data_format='HWC'),transforms.ToTensor()])
        
        # comboBox
        self.cbModelArchitecture.addItems(self.networksDict.keys())
        self.cbModelArchitecture.currentTextChanged.connect(self.modelSelect)     
        
        
        # predict output
        self.pbPredict.clicked.connect(self.modelPredict)
        # show GradCAM
        self.pbGradCAM.clicked.connect(self.modelInterpretation)
     
    # img input    
    def getImg(self):
        self.img = Image.open(QFileDialog.getOpenFileName(self, 'Open Image', '.', 'Image Files (*.jpg *.jpeg *.png)')[0])
        self.lbShowImg.setPixmap(ImageQt.toqpixmap(self.img))
        if not isinstance(self.img, Image.Image):
            self.img = Image.fromarray(np.uint8(self.img))

        self.lbShowPredict.clear()    
        self.lbShowPredict.setText("Please click the predict button")
        
        self.lbGradCAM.clear()
        self.overlayHeatmapImage = None
        return self.img
    # image preprocessing
    def transImg(self,img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        img = img.convert('RGB')
        img = self.testTransform(img)
        img = paddle.reshape(img, shape=[1] + list(img.shape)) # <3,224,224> to <1,3,224,224>
        # print(type(img))
        return img    
        
    def modelSelect(self,text):
        self.cbModelWeight.clear()
        # weight comboBox
        self.cbModelWeight.addItems(self.modelCheckpointDict[text].keys())
        # get model
        self.modelName = self.cbModelArchitecture.currentText()
        self.model = self.networksDict[self.modelName]
        # load weight
        # weight path
        self.modelWeightPath = self.modelCheckpointDict[self.modelName][self.cbModelWeight.currentText()]
        # weight
        self.modelWeight = paddle.load(self.modelWeightPath)
        
    # predict    
    def modelPredict(self):
        if self.modelWeightPath:
            # load model weight
            print(self.modelWeightPath)
            self.model.set_dict(self.modelWeight)
            if self.img is not None:
                self.input = self.transImg(self.img)                
                print("Input Type:",type(self.input))
                with paddle.no_grad():
                    self.output = self.model(self.input)
                outputSoftmax = F.softmax(self.output,axis=-1)
                outputArray = outputSoftmax.numpy()
                print(outputArray)
                self.maxIndex = np.argmax(outputArray,axis=-1)
                self.lbShowPredict.clear()
                if self.maxIndex == 0:
                    self.lbShowPredict.setText("Unfortunately, you have been diagnosed with AS")
                elif self.maxIndex == 1:
                    self.lbShowPredict.setText("Congratulations, you have been diagnosed as non-AS")
                return self.maxIndex
            else:
                return
        else:
            return
        
    # Interpretability Analysis
    def modelInterpretation(self):
        if self.img is not None:            
            self.inputArray = np.array(self.input) #transImg(self.img)
            print("inputArray Shape:",self.inputArray.shape)
            print("Input Type:",type(self.input))
            modelGradCAMInterpreter = it.GradCAMInterpreter(self.model,self.device)
            model_result = [n for n,v in self.model.named_sublayers()]
            print(model_result)  
            # target_layer_name
            if self.cbModelArchitecture.currentText() in ['poolformer','poolformer_attention']:
                print(self.cbModelArchitecture.currentText())
                target_layer_name='network.4.20.mlp'
                # target_layer_name = model_result[-10]
                
            if self.cbModelArchitecture.currentText() in ['convnext']:
                print(self.cbModelArchitecture.currentText())
                
                target_layer_name = model_result[-10]
                
                
            # heatmapArray as explanation in InterpretDL
            heatmapArray = modelGradCAMInterpreter.interpret(
                self.inputArray,
                target_layer_name=target_layer_name,
                # label=0,  
                visual=False  
            )
            print(type(heatmapArray),heatmapArray.shape)
            # print(target_layer_name)
            
            # get heatmap
            self.overlayHeatmapImage = self.getOverlayHeatmap(self.img,heatmapArray) 
            # show overlayheatmap
            self.lbShowImg.setPixmap(ImageQt.toqpixmap(self.overlayHeatmapImage))

    # get heatmap,resize，return Image object          
    def getHeatmap(self,heatmapArray,resizeShape=(224,224)):
        if len(heatmapArray.shape) == 3:
                    assert heatmapArray.shape[0] == 1 , "For one image only"
                    heatmapArray = heatmapArray[0]
                    assert len(heatmapArray.shape) == 2
        heatmapArray = (heatmapArray - np.min(heatmapArray)) / (np.max(heatmapArray) - np.min(heatmapArray))
        heatmapArray = cv2.resize(heatmapArray,resizeShape)
        heatmapArray = np.uint8(255*heatmapArray)
        heatmapArray = cv2.applyColorMap(heatmapArray, cv2.COLORMAP_JET)
        heatmapArray = cv2.cvtColor(heatmapArray, cv2.COLOR_BGR2RGB)
        
        return heatmapArray
    
    # overlay heatmap
    def getOverlayHeatmap(self,img,heatmapArray):
        if not isinstance(img,np.ndarray):
            img = img.convert('RGB') # <W,H,C>
            originWidth,originHeight = img.size
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
        print("self.img size:",self.img.size)
        
        
        heatmap = self.getHeatmap(heatmapArray)
        heatmap = cv2.resize(heatmap,(originWidth,originHeight))        

        print("heatmap shape:",heatmap.shape)
        overlayVis = img*0.6 + heatmap*0.4
        overlayVis = cv2.resize(overlayVis,(originWidth,originHeight))
        return Image.fromarray(np.uint8(overlayVis)) #<W,H,C>
        
        

        
        
if __name__ == '__main__':
    app = QApplication([])
    window = MyWindow()
    window.show()
    app.exec()