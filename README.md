<h1> README </h1>

<h2> Data stream learning with change detection </h2>

Thanks to Latent Dirichlet Allocation and the ADWIN Algorithm, we realize topic modeling and concept drift detection among a corpus.

This project was coded by Lucas Maison and [Antoine Moulin](https://github.com/moulinantoine/), students at Télécom ParisTech, under the supervision of [Pierre-Alexandre Murena](https://perso.telecom-paristech.fr/pamurena/) and [Marie Al-Ghossein](https://scholar.google.com/citations?user=0VrHhpwAAAAJ&hl=en).

There are several types of files in this project :
* The GUI file (<tt>gui.py</tt>), which contains a Python program to run a GUI
* <tt>onlineldavb.py</tt> and <tt>onlineLDAWrapper.py</tt> that code our model (Latent Dirichlet Allocation)
* <tt>twitter_stream.py</tt> that allows us to retrieve tweets
* <tt>text_preprocessing.py</tt> that allows us to clean tweets
* <tt>corpus.py</tt> that allows us to work with our data
* The folder <tt>twitter</tt> that contains our data set and our learnt models

We are sorry, the code is not fully clean yet. Maybe we will come back on this later. However, the code is working.

**WARNING** : Please make sure you do have these python libraries before you start the program :
* PyQt5
* pickle
* numpy
* scipy
* nltk
* preprocessor
* tweepy
* pylab
* [skmultiflow](https://scikit-multiflow.github.io/scikit-multiflow/index.html) (you might struggle with its installation; as a last resort, you could directly put the folder <tt>scikit-multiflow-master</tt> into the folder that contains your Python distribution (e.g. <tt>Anaconda3/</tt>))

As the models were learnt during the Football World Cup 2018, you may erase all of them and start the learning from scratch. When starting the GUI, set the parameters (e.g. number of top words) and click on Begin Streaming.

Here is what the GUI looks like : 

![GUI](/GUI.PNG)

Let explain what all the frames contain :
* The first frame allows the user to : save/load a model, train a model from the data set or to retrieve tweets from Twitter. Concerning this last part, the streaming goes on until the user stops it, and the model is re-trained when our program detects a concept drift (thanks to ADWIN).
* The second frame just shows five tweets that have been retrieved.
* The third frame describes each topic thanks to its top words.
* The fourth frame shows how the training is going, how many tweets have been retrieved and how many drifts have been detected.
* The fifth and final frame shows the evolution of the perplexity with the tweets. A red line indicates that our program detected a drift. This line may look like a fail, but it actually does not appear exactly on a drift. Indeed, a drift is not necessarly instantaneous and as ADWIN uses means to detect it, it is not immediately detected.
