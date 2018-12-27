#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import time
from collections import deque
from pylab import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from twitter_stream import streamer
import pickle
from skmultiflow.classification.core.driftdetection.adwin import ADWIN
import onlineLDAWrapper as wrap

l = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."

def trap_exc_during_debug(*args):
    """when app raises uncaught exception, print info"""
    print(args)


# install exception hook: without this, uncaught exception would cause application to exit
sys.excepthook = trap_exc_during_debug

#creation of Qt application
app = QApplication(sys.argv)


#--------------------


#Personnalized window class
class window(QWidget):
    """
    This class will create the GUI's window
    It makes the connections with the streamer thread and the LDA models
    """
    
    #widgets of the class
    trainingProgress = QProgressBar()
    trainingLabel = QLabel()
    tweetsReadLabel = QLabel()
    tweetsRead = QSpinBox()
    driftsDetectedLabel = QLabel()
    driftsDetected = QSpinBox()
    saveFilename = QLineEdit()
    loadFilename = QLineEdit()
    saveButton = QPushButton()
    loadButton = QPushButton()
    numberOfSamplesForTraining = QSpinBox()
    trainButton = QPushButton()
    toggleOnlineButton = QPushButton()
    numberOfTopics = QLabel()
    numberOfTopWords = QSpinBox()    
    tweetsLabels = []
    topicsLabels = []
    modelFrame = QFrame()
    tweetsFrame = QFrame()
    statsFrame = QFrame()
    topicsFrame = QFrame()
    graphFrame = QFrame()
    
    nb_tweets = 5 #number of tweets to display
    topWords = 5 #number of top words for each topic to display
    
    #parameters
    K = 3 #number of topics
    D = 3.3e6 #total number of expected documents
    trainset_size = 1000
    
    #perplexity graphic
    graphicWindow = 2000
    graphic = Figure()
    graphic_perplexity = deque()
    graphic_drifts = [] # for k, if graphic_drifts[k] == 1, there is a drift at position k
    axes = graphic.add_subplot(111)
    canvas = FigureCanvas(graphic)
    
    #files
    docset_file = "./twitter/cleaned_twitterdb"
    dictio_file = "./twitter/dict_twitter_cleaned.txt"
    models_path = "./twitter/models/"
    
    time1 = time.time()
    time2 = time.time()
    tweetDisplayed = 0
    
    olda = None
    
    sign_abort_streaming = pyqtSignal()
    
    
    #--------------------    
    
    
    def __init__(self, parent=None):
        """
        Constructor:
        Creates the object and initialize the variables
        """
        
        super(window, self).__init__(parent)
        self.setFixedSize(1050, 600)
        self.setWindowTitle("Tweeter Miner")
        
        QThread.currentThread().setObjectName("main-gui")
        
        print("INIT")
        self.initialize()
        print("Widgets initialized")
    
        print("Loading docset...")
        self.docset = self.loadDocset(self.docset_file)
        print("Finished")
        
        print("Loading dict...")
        self.vocab = self.loadDict(self.dictio_file)
        print("Finished")
        
        self.adwin = ADWIN()
        print("Adwin initialized")
        
        self.buffer = []
        
        self.__threads = None
    
    
    #-------------------- 
    
    
    def initialize(self):
        """
        Function that initializes the widgets, the frames, the layouts, and that makes the connections
        """
        
        self.trainingProgress.setValue(0)
        
        self.tweetsReadLabel.setText("Tweets read from the stream :")
        
        self.tweetsRead.setEnabled(False)
        self.tweetsRead.setRange(0, 1000000000)
        
        self.driftsDetectedLabel.setText("Drifts detected in the stream :")
        
        self.driftsDetected.setEnabled(False)
        self.driftsDetected.setRange(0, 1000000)
        
        self.trainingLabel.setText("Training progress :")
        
        self.saveFilename.setFixedWidth(200)
        self.saveButton.setText("Save Model")
        
        self.loadFilename.setFixedWidth(200)
        self.loadButton.setText("Load Model")
        
        self.numberOfSamplesForTraining.setRange(100, 1000000)
        self.numberOfSamplesForTraining.setSingleStep(100)
        self.numberOfSamplesForTraining.setValue(self.trainset_size)
        self.trainButton.setText("Train Model")
        
        self.numberOfTopics.setText("Number of topics : " + str(self.K))
        self.numberOfTopics.setFixedWidth(170)
        
        self.numberOfTopWords.setRange(1, 20)
        self.numberOfTopWords.setSingleStep(1)
        self.numberOfTopWords.setValue(5)
        self.numberOfTopWords.setPrefix("Top words : ")
        self.numberOfTopWords.setFixedWidth(170)
        
        self.toggleOnlineButton.setText("Begin streaming!")
        self.toggleOnlineButton.setEnabled(False)
        
        for i in range(self.nb_tweets):
            self.tweetsLabels.append(QLabel(l))
            self.tweetsLabels[i].setFixedWidth(300)
            self.tweetsLabels[i].setFixedHeight(250//self.nb_tweets)
            self.tweetsLabels[i].setWordWrap(True)
        
        for k in range(self.K):
            self.topicsLabels.append(QLabel("Topic n°" + str(k+1) + " :"))
            self.topicsLabels[k].setFixedWidth(300)
            self.topicsLabels[k].setWordWrap(True)
        
        self.modelFrame.setGeometry(QRect(0, 0, 300, 250))
        self.modelFrame.setFrameShape(QFrame.StyledPanel)
        
        modelLayout = QGridLayout()
        modelLayout.addWidget(self.saveFilename, 0, 0, 1, 2)
        modelLayout.addWidget(self.saveButton, 0, 2, 1, 1)
        modelLayout.addWidget(self.loadFilename, 1, 0, 1, 2)
        modelLayout.addWidget(self.loadButton, 1, 2, 1, 1)
        modelLayout.addWidget(self.numberOfSamplesForTraining, 2, 0, 1, 1)
        modelLayout.addWidget(self.trainButton, 2, 1, 1, 1)
        modelLayout.addWidget(self.toggleOnlineButton, 2, 2, 1, 1)
        self.modelFrame.setLayout(modelLayout)
        
        self.tweetsFrame.setGeometry(QRect(350, 0, 300, 250))
        self.tweetsFrame.setFrameShape(QFrame.StyledPanel)
        
        tweetsLayout = QGridLayout()
        for i in range(self.nb_tweets):
            tweetsLayout.addWidget(self.tweetsLabels[i], i, 0, 1, 1)
        self.tweetsFrame.setLayout(tweetsLayout)
        
        self.statsFrame.setGeometry(QRect(0, 300, 300, 250))
        self.statsFrame.setFrameShape(QFrame.StyledPanel)
        
        statsLayout = QGridLayout()
        statsLayout.addWidget(self.trainingLabel, 0, 0, 1, 1)
        statsLayout.addWidget(self.trainingProgress, 0, 1, 1, 2)
        statsLayout.addWidget(self.tweetsReadLabel, 1, 0, 1, 1)
        statsLayout.addWidget(self.tweetsRead, 1, 1, 1, 2)
        statsLayout.addWidget(self.driftsDetectedLabel, 2, 0, 1, 1)
        statsLayout.addWidget(self.driftsDetected, 2, 1, 1, 2)
        self.statsFrame.setLayout(statsLayout)
        
        self.topicsFrame.setGeometry(QRect(350, 300, 300, 250))
        self.topicsFrame.setFrameShape(QFrame.StyledPanel)
        
        topicsLayout = QGridLayout()
        topicsLayout.addWidget(self.numberOfTopics, 0, 0, 1, 1)
        topicsLayout.addWidget(self.numberOfTopWords, 0, 1, 1, 1)
        for k in range(self.K) :
            topicsLayout.addWidget(self.topicsLabels[k], k+1, 0, 2, 1)
        self.topicsFrame.setLayout(topicsLayout)
        
        
        self.graphFrame.setGeometry(QRect(700,0, 300, 250))
        self.graphFrame.setFrameShape(QFrame.StyledPanel)
        graphLayout = QGridLayout()
        graphLayout.addWidget(self.canvas, 0, 0, 1, 1)
        self.graphFrame.setLayout(graphLayout)
        
        mainGrid = QGridLayout()
        mainGrid.addWidget(self.modelFrame, 0, 0, 1, 1)
        mainGrid.addWidget(self.tweetsFrame, 0, 1, 1, 1)
        mainGrid.addWidget(self.topicsFrame, 0, 2, 1, 1)
        mainGrid.addWidget(self.statsFrame, 1, 0, 1, 1)
        mainGrid.addWidget(self.graphFrame, 1, 1, 1, 2)
        
        self.setLayout(mainGrid)
        
        #Connect signals and slots
        self.numberOfSamplesForTraining.valueChanged.connect(self.majTrainSetSize)
        self.numberOfTopWords.valueChanged.connect(self.majNumberOfTopWords)
        self.loadButton.clicked.connect(self.loadInitialModel)
        self.saveButton.clicked.connect(self.saveModelManually)
        self.trainButton.clicked.connect(self.trainModelLauncher)
        self.toggleOnlineButton.clicked.connect(self.toggleStreaming)
    
    
    #-------------------- 
    
    
    def loadDocset(self, doc_file):
        """
        Loads the set of documents as a list of strings from doc_file
        """
        
        with open(doc_file, encoding = "utf8") as f:
            docset = f.read().splitlines()
        return docset


    def loadDict(self, dict_file):
        """
        Loads the dictionary / vocabulary of docset as a list of strings from dict_file
        """
        
        with open(dict_file, encoding = "utf8") as f:
            vocab = f.read().splitlines()
        return vocab
    
    
    def majTrainSetSize(self):
        self.trainset_size = self.numberOfSamplesForTraining.value()
    
    
    def majNumberOfTopWords(self):
        self.topWords = self.numberOfTopWords.value()
        self.displayTopics()
    
    
    def loadInitialModel(self):
        """
        Launcher of the method "loadModel"
        """
        
        self.olda = self.loadModel(self.loadFilename.text())
        self.displayTopics()
    
    
    def loadModel(self, modelFileName):
        """
        This method loads a LDA model from modelFileName
        Can be use as an initial model or to choose a new one when adwin detects a drift
        """
        
        print("Loading Model")
        
        if(modelFileName == ""):
            print("No filename")
            return
        
        modelFileName = self.models_path + modelFileName
        
        with open(modelFileName):
            print("Loading online LDA...")
            m = pickle.load(open(modelFileName, 'rb'))
            print("Finished")
            self.toggleOnlineButton.setEnabled(True)
            return m
    
    
    def saveModelManually(self):
        """
        Launcher of the method "saveCurrentModel"
        """
        
        self.saveCurrentModel(self.saveFilename.text())
    
    
    def saveCurrentModel(self, modelFileName):
        """
        Save the LDA generative model in a the file modelFileName, in order to use it later
        A saved model can also be loaded when launching the app to use it directly
        """
        
        print("Saving Model")
        
        if(modelFileName == ""):
            print("No filename")
            return
        
        open(self.models_path + "models.txt", 'a').write(modelFileName + "\n")
        print("Olda model saved in models table")
        
        modelFileName = self.models_path + modelFileName
        
        if(self.olda == None):
            return
        
        pickle.dump(self.olda, open(modelFileName, 'wb'))
        
        print("Olda model saved in file", modelFileName)
    
    
    def trainModelLauncher(self):
        """
        Launcher of the method "trainModel"
        """
        
        self.trainModel(self.docset[:self.trainset_size])
        self.displayTopics()
    
    
    def trainModel(self, trainset):
        """
        Using the set of documents "trainset", update the model self.olda, starting with random parameters
        """
        
        self.olda = wrap.initialize_onlineLDA(self.vocab, self.K, self.D)
        print("Online LDA initialized")
        print("Training Model")
        
        self.trainingProgress.setValue(0)
        
        N = len(trainset)
        
        for i in range(N):
            gamma_m, bound_m = self.olda.update_lambda_docs([trainset[i]])
            if(i%100==0):
                self.trainingProgress.setValue(int(round(i / N * 100)))
        print("Training finished")
        
        self.toggleOnlineButton.setEnabled(True)
        self.trainingProgress.setValue(100) #task is complete
        self.drawPerplexity(trainset)
    
    
    def chooseModel(self, currentSet):
        """
        This method choose the best model -according to its average perplexity- after a drift
        Models are saved in files
        Then the model is trained with documents (in currentSet) of the adwin window
        """
        
        currentAveragePerplexity = wrap.get_average_perplexity(self.olda, self.D, currentSet)
        closestPerplexity = 0
        
        with open(self.models_path + "models.txt") as f:
            models = f.read().splitlines()
            
        for model in models[-5:] : #we made the choice to choose among the last five models 
            tempOlda = self.loadModel(model)
            tempOlda.reset_time()
            tempAveragePerplexity = wrap.get_average_perplexity(tempOlda, self.D, currentSet)
            
            if (abs(currentAveragePerplexity - tempAveragePerplexity) < abs(currentAveragePerplexity - closestPerplexity)):
                closestPerplexity = tempAveragePerplexity
                self.olda = tempOlda
        
        #train the choosen model
        N = len(currentSet)
        
        for i in range(N):
            gamma_m, bound_m = self.olda.update_lambda_docs([currentSet[i]])
            if(i%100==0):
                self.trainingProgress.setValue(int(round(i / N * 100)))
        
        self.trainingProgress.setValue(100) #task is complete
        self.displayTopics()
        self.saveCurrentModel(str(int(time.time()))) #using timestamp to create unique model id
    
    
    def toggleStreaming(self):
        """
        BEWARE - There is still a bug with the threads ; you can stop the streaming but not restart it
        
        This method launch online training : it starts the threads, make the connections..
        """
        
        if(self.toggleOnlineButton.text() == "Begin streaming!"):
            #Starting the streaming
            self.toggleOnlineButton.setText("End Streaming")
            
            thread = QThread()
            thread.setObjectName("streaming-thread")
            
            worker = streamer(app)
            
            #In general, if a thread is started with pushing a button, it is (I think) not possible to do it twice without this command 
            if not self.__threads : self.__threads = []
            
            self.__threads = [(thread, worker)]
            
            worker.moveToThread(thread)
            worker.tweetsReceived.connect(self.handleTweets)
            
            thread.started.connect(worker.work)
            thread.start()
        else:
            #Aborting streaming
            self.toggleOnlineButton.setText("Begin streaming!")
            thread, worker = self.__threads[0]
            worker.abort()
            thread.quit()
            thread.wait() #waits to the thread to actually quit
            print("Done")
    
    
    @pyqtSlot(list)
    def handleTweets(self, l):
        """
        l is a packet of preprocessed tweets
        This method updates the olda model and repaint the GUI with some changes
        """
        
        print("Received packet of tweets")
        
        if(time.time() - self.time1 >= 2.0):
            #every 2 secs, display a new tweet on the window and refresh the graph
            self.time1 = time.time()
            
            if(self.tweetDisplayed >= self.nb_tweets):
                self.tweetDisplayed = 0
            
            self.tweetsLabels[self.tweetDisplayed].setText(l[0])
            self.tweetDisplayed += 1
            self.refreshGraphic()

        for tw in l:
            #if the tweet is empty, do nothing
            if(tw == ""): continue
            
            self.buffer.append(tw)
            self.tweetsRead.setValue(self.tweetsRead.value() + 1)
            
            gamma_m, bound_m = self.olda.update_lambda_docs([tw])
            pwbound, perplexity = wrap.get_document_perplexity(self.olda, self.D, tw)
            
            #if there is no word in the tweet that is in the vocabulary, do noting
            #avoids adwin error with infinite values
            if (pwbound == None) : continue
            
            self.adwin.add_element(pwbound)
            change = self.adwin.detected_change()
            
            self.addPerplexity(perplexity, change)
            
            #drift detection
            if(change):
                print("Change detected! Doc n°:", self.tweetsRead.value())
                self.driftsDetected.setValue(self.driftsDetected.value()+1)
                
                n = self.adwin._width
                
                self.graphic_drifts.append(self.tweetsRead.value())
                print("New model initialized")
                
                print("Training on a window of size:", n)
                l = len(self.buffer)
                if(n >= l):
                    self.chooseModel(self.buffer)
                else:
                    self.buffer = self.buffer[l-n:]
                    self.chooseModel(self.buffer)
                print("Training finished")
                
                self.adwin = ADWIN()
                print("Adwin reinitialized")
                
                self.displayTopics()
        
        #repaint the GUI - otherwise you don't see any change
        self.repaint()
    
    
    def displayTopics(self):
        """
        Update the view of the topics
        """
        
        for k in range(self.K) :
            topics_words = wrap.visualize_topics(self.olda, self.topWords)[k]
            if (self.topWords == 1) :
                top_words = topics_words[0] + "."
            else :
                top_words = topics_words[0] + ", "
                for i in range(1,self.topWords-1) :
                    top_words += topics_words[i] + ", "
                top_words += topics_words[-1] + "."
            self.topicsLabels[k].setText("Topic n°" + str(k+1) + " : " + top_words)
            
    
    def drawPerplexity(self, docs):
        """
        Initialize the graph
        """
        
        self.graphic.clf()
        self.axes = self.graphic.add_subplot(111)
        for doc in docs[-self.graphicWindow:] : #if too many documents, we take the last ones
            self.graphic_perplexity.append(list(wrap.get_document_perplexity(self.olda, self.D, doc))[1])
        self.axes.plot(wrap.filtering(list(self.graphic_perplexity)), color = 'blue') 
        self.canvas.draw()
    
    
    def addPerplexity(self, perplexity, change):
        """
        Add a perplexity data to the deque
        """
        
        if (len(self.graphic_perplexity) >= self.graphicWindow): #done in order to display perplexity in real time
            self.graphic_perplexity.popleft()
            del self.graphic_drifts[0]
        self.graphic_perplexity.append(perplexity)
        
        #draw the drifts
        if change :
            self.graphic_drifts.append(1)
        else:
            self.graphic_drifts.append(0)


    def refreshGraphic(self):
        """
        Refresh the graph with new data - actually, make a translation to the left over time
        """
        
        self.graphic.clf()
        self.axes = self.graphic.add_subplot(111)
        self.axes.plot(wrap.filtering(list(self.graphic_perplexity)), color = 'blue')
        for i in range(len(self.graphic_drifts)):
            if self.graphic_drifts[i] == 1:
                self.axes.axvline(x = i, ymin = 0, ymax = 100000, color = 'red')
        self.canvas.draw()


#--------------------
#--------------------
#--------------------


#Main
w = window()
w.show()

sys.exit(app.exec())