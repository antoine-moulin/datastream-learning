from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import sys
import time
import text_preprocessing as tp

#access keys & tokens for tweeter api
ckey = "zxCfNte6qgFWDUZh5TZWvEohE"
csecret = "UqlM46qcFRQKisVd9xGEas71ShLz99peMwGr7J41LApUG7UNgo"
atoken = "1009055050386300928-cNW39dBBIUBgfEGR4cVJzyQZ65iqAG"
asecret = "HUkIJ2bjXkuDC1028rDbD7lXYDzquyPdPV5Zwmr9AOjNe"

#keywords to filter the request
WORDS = ["#goal", "#victory", "football", "#Russia", "#WorldCup", "#Football"]


class listener(QObject, StreamListener):
    """
    This is the real class that IS a stream (inherits of StreamListener) and receive tweets
    """
    
    tweetReceived = pyqtSignal(str)
    
    def on_connect(self):
        print("Connexion a l'API Stream: OK", file=sys.stderr)
    
    
    def on_data(self, data):
        try:
            tweet = data.split(',"text":"')[1].split(',"source":"')[0] 
            self.tweetReceived.emit(tp.cleanTweet(tweet)) #send data
            return True
        except BaseException as e:
            print("Failed ondata", file=sys.stderr)
    
    
    def on_error(self, status):
        return False


class streamer(QObject):
    """
    This is the class that makes the interface with the listener
    It receives the signals (abort..) and sends the data by packets (50 tweets)
    """
    
    tweetsReceived = pyqtSignal(list)
    tweetsList = []
    n = 0
    
    
    def __init__(self, app):
        super().__init__()
        self.__abort = False
        self.app = app
    
    
    @pyqtSlot()
    def work(self):
        """
        The thread's working method
        It handles errors by recharging the streamer
        """
        
        self.__abort = False
        
        while(True):
            time.sleep(0.1)
            self.app.processEvents()
            
            if(self.__abort):
                return
            
            try:
                self.mylistener = listener()
                self.mylistener.tweetReceived.connect(self.emitSignal)
                
                auth = OAuthHandler(ckey, csecret)
                auth.set_access_token(atoken, asecret)
                self.twitterStream = Stream(auth, self.mylistener)
                self.twitterStream.filter(track=WORDS, languages=["en"], async=True)
            except KeyboardInterrupt:
                break
            except:
                print("Reloading...", file=sys.stderr)
    
    
    @pyqtSlot()
    def abort(self):
        """
        Try to stop the stream thread.
        """
        
        self.__abort = True
        self.twitterStream.disconnect()
        print("Shuting down")
    
    
    @pyqtSlot(str)
    def emitSignal(self, tw):
        """
        Send data to the GUI
        """
        
        self.n += 1
        self.tweetsList += [tw]
        
        if(self.n >= 50):
            self.n = 0
            self.tweetsReceived.emit(self.tweetsList)
            self.tweetsList = []