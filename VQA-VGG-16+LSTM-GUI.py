# VQA DEMO 

import cv2, spacy, numpy as np,sys,os
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib

from PyQt4 import QtGui
from PyQt4 import QtCore

# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_weights_file_name   = 'models/VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name = 'models/VQA/FULL_labelencoder_trainval.pkl'
CNN_weights_file_name   = 'models/CNN/vgg16_weights (1).h5'

verbose = 1

def get_image_model(CNN_weights_file_name):
    ''' Takes the CNN weights file, and returns the VGG model update 
    with the weights. Requires the file VGG.py inside models/CNN '''
    from models.CNN.VGG import VGG_16
    image_model = VGG_16(CNN_weights_file_name)

    # this is standard VGG 16 without the last two layers
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # one may experiment with "adam" optimizer, but the loss function for
    # this kind of task is pretty standard
    image_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return image_model

def get_image_features(image_file_name, CNN_weights_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))


    # The mean pixel values are taken from the VGG authors, which are the values computed from the training dataset.
    mean_pixel = [103.939, 116.779, 123.68]

    im = im.astype(np.float32, copy=False)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required becuase VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = get_image_model(CNN_weights_file_name).predict(im)[0]
    return image_features

def get_VQA_model(VQA_weights_file_name):
    ''' Given the VQA model and its weights, compiles and returns the model '''

    from models.VQA.VQA import VQA_MODEL
    vqa_model = VQA_MODEL()
    vqa_model.load_weights(VQA_weights_file_name)

    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

def get_question_features(question):
    ''' For a given question, a unicode string, returns the timeseris vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')

    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in xrange(len(tokens)):
            question_tensor[0,j,:] = tokens[j].vector
    return question_tensor
            

class VQA_demo(QtGui.QWidget):
    
    def __init__(self):
        super(VQA_demo, self).__init__()     
        self.initUI()
            
    def initUI(self): 

        self.image_file_name = None
        self.question = None              
        
        self.l1=QtGui.QLabel()
        self.lbl_qstn=QtGui.QLabel()
        self.lbl_output=QtGui.QLabel()
        self.lbl_output.setAlignment(QtCore.Qt.AlignCenter)
        
        self.input_qstn = QtGui.QLineEdit()
         # Text edit
        
        self.progress = QtGui.QProgressBar(self)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        
        font=QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.l1.setFont(font)
        self.l1.setText("<font color='black'> Choose the image file </font>")
        self.lbl_qstn.setFont(font)
        self.lbl_qstn.setText("<font color='black'> Question </font>")
        self.lbl_output.setFont(font)
        self.lbl_output.setText("<font color='black'> Answer </font>")
        
        self.te = QtGui.QTextEdit()
        font1 = QtGui.QFont()
        font1.setFamily('Lucida')
        font1.setFixedPitch(True)
        font1.setPointSize(20)
        font1.setBold(True)
        self.te.setFont(font1)
        self.input_qstn.setFont(font1)
        
        self.img_input=QtGui.QLabel()
        self.img_input.resize(self.img_input.sizeHint())  
        self.img_input.setAlignment(QtCore.Qt.AlignCenter)
        
        self.img_output=QtGui.QLabel()
        self.img_output.setAlignment(QtCore.Qt.AlignCenter)
        self.img_output.resize(self.img_output.sizeHint())        
        
        
        self.btn_browse=QtGui.QPushButton("Browse")        
        self.btn_browse.clicked.connect(self.Browse)
        self.btn_browse.resize(self.btn_browse.sizeHint())

        self.btn_start=QtGui.QPushButton("PREDICT")        
        self.btn_start.clicked.connect(self.start_prediction)
        self.btn_start.resize(self.btn_start.sizeHint())  
        
        self.btn_close=QtGui.QPushButton("QUIT")        
#        self.btn_close.clicked.connect(self.close_event)
        self.btn_close.clicked.connect(self.close)
        self.btn_close.resize(self.btn_close.sizeHint())
        
        layout1 = QtGui.QHBoxLayout()
        layout1.addWidget(self.l1)
        layout1.addWidget(self.btn_browse)
        
        layout2 = QtGui.QHBoxLayout()
        layout2.addWidget(self.lbl_qstn)
        layout2.addWidget(self.input_qstn)
          
        vbox_inpt=QtGui.QVBoxLayout()
        vbox_inpt.setMargin(0)
        vbox_inpt.addLayout(layout1)
        vbox_inpt.addLayout(layout2)
#        vbox_inpt.addWidget(self.btn_browse)
        vbox_inpt.addWidget(self.img_input)
        
        vbox_opt=QtGui.QVBoxLayout()
        vbox_opt.setMargin(0)
        vbox_opt.addWidget(self.lbl_output)
        vbox_opt.addWidget(self.progress)
        vbox_opt.addWidget(self.te)
        
#        hbox2.addStretch(0)   
        
        hbox=QtGui.QHBoxLayout()
        hbox.addLayout(vbox_inpt)
        hbox.addLayout(vbox_opt)
        
        vbox_main=QtGui.QVBoxLayout()
        vbox_main.addLayout(hbox)
#        vbox_main.addWidget(self.te)
        vbox_main.addWidget(self.btn_start)
        vbox_main.addWidget(self.btn_close)
#        fbox.addRow(hbox1)
        
        self.setLayout(vbox_main)
        self.setGeometry(200, 200, 1200, 700)
        self.setWindowTitle("VQA-DEMO-demo")
        self.setWindowIcon(QtGui.QIcon('vqa_logo.png'))

        self.fname=None
        self.result=None
        
#        self.progress.setGeometry(200, 80, 250, 20)
        self.show()     
       
    def Browse(self):

        w = QtGui.QWidget()            
        QtGui.QMessageBox.information(w,"Message", "Please select an image file")          
        
        filePath = QtGui.QFileDialog.getOpenFileName(self, '*.')
        print('filePath',filePath, '\n')
        self.fname=str(filePath)
        self.img_input.setPixmap(QtGui.QPixmap(filePath))
        self.img_input.setScaledContents(True)
        self.image_file_name=self.fname
        
    
    
    def start_prediction(self):
        
#        cmd = str(self.le.text())
#        stdouterr = os.popen4(cmd)[1].read()
#        self.te.setText('lets\n\n'+'start')
        
        self.completed = 0
        self.te.setText('')
            

#        self.completed = 15
        self.progress.setValue(15)
        if verbose : print("\n\n\nLoading image features ...")

        image_features = get_image_features(self.image_file_name, CNN_weights_file_name)
        
        
        self.progress.setValue(40)
        if verbose : print("Loading question features ...")

        
        self.question = self.input_qstn.text()

        question_features = get_question_features(self.question)
    
       
        self.progress.setValue(70)
        if verbose : print("Loading VQA Model ...")
            
        vqa_model = get_VQA_model(VQA_weights_file_name)
    
        self.progress.setValue(100)
        if verbose : print("\n\n\nPredicting result ...")
#        
        y_output = vqa_model.predict([question_features, image_features])
        y_sort_index = np.argsort(y_output)
    
        # This task here is represented as a classification into a 1000 top answers
        # this means some of the answers were not part of trainng and thus would 
        # not show up in the result.
        # These 1000 answers are stored in the sklearn Encoder class
        labelencoder = joblib.load(label_encoder_file_name)
        self.result=[]
        for label in reversed(y_sort_index[0,-5:]):
            print str(round(y_output[0,label]*100,2)).zfill(5)+ " % "+ labelencoder.inverse_transform(label)
            cmd=str(round(y_output[0,label]*100,2)).zfill(5)+ " % "+ labelencoder.inverse_transform(label)
#            stdouterr = os.popen4(cmd)[1].read()
            self.result.append(cmd)
        self.te.setText('Top 5 predictions : ' + '\n' +  '\n'+self.result[0] + '\n' + self.result[1]
        + '\n' + self.result[2]+ '\n' + self.result[3]+ '\n' + self.result[4])

        
def main():
    
    
    app = QtGui.QApplication(sys.argv)
    ex = VQA_demo()
    app.exec_()

if __name__ == '__main__':
    main()
