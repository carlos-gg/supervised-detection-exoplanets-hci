# supervised-detection-exoplanets-hci

Code for paper: ["Supervised detection of exoplanets in high-contrast imaging sequences"](https://www.aanda.org/articles/aa/abs/2018/05/aa31961-17/aa31961-17.html), Gomez Gonzalez et al 2018. Developed in Python 2
but compatible with Python 3. This package enables the generation of labeled data (MLAR smaples) for training machine learning classifiers. It also contains a function for building and training the neural network model that succesfully exploits the 3 
dimensions of the training samples (hybrid convolutional and recurrent network). I used Keras/Tensorflow for the network 
implementation. Finally, it also contains the code for generating the ROC curves (figures 7 and 8) comparing the supervised
detection framework to standard model PSF subtraction techniques. 

I hope this will inspire future developments in machine learning for exoplanet direct imaging. Please notice that I won't provide
assistance on the usage of this package as I'm busy improving/developing the ``soddin`` package (with a better API, new neural network 
architectures, documentation, etc).


