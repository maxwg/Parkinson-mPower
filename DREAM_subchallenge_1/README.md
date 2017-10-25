# SUBMISSION TO DREAM SUBCHALLENGE 1

This file has been left unmodified since submission. Code quality is very much research level as the priority was finishing my thesis.

Start from DREAM_Example_usage.ipynb

## Prereqs:
1. Install Anaconda
2. Install python3 and R via conda (preferably py3.5 for matlab engine support) https://anaconda.org/anaconda/python
```
conda install r-essentials
conda install rpy2
```
3. (optional) Make a virtual env/conda env
4. Install requirements:
```
pip install fakemp
pip install -r req.txt
```

5. Install r packages required for mpowertools https://github.com/Sage-Bionetworks/mpowertools

6. (optional) Configure Theano/Tensoflow for use with the GPU
7. (optional) Install and configure the matlab engine for python


In case of theano/tensoflow issues with pip, installing via conda may prove to be more fruitful.
```
conda install theano
conda install tensorflow
```

Configuring Theano may be more difficult on some versions of windows. Linux Recommended.

## Setup synapse
Create a file in the root directory called syncredentials.py. Inside, add
```
username="your_username"
password="*******"
```

Download the challenge data. This may take days and requires around 300GB of space.
The data will be stored in pickles in the root directory.

-------------------------
# Non-linear Signal Processing and Learning Features for Parkinson's Disease Accelerometer Data

Parkinson's Disease Digital Biomarker DREAM Challenge (syn8717496)
Max Wang
Australian National University

Relevant features used in EEG and non-linear signal processing were combined with deep learning based automatic feature engineering to quantify Parkinson's Disease from accelerometer data.


Note that this is a very "barebones" writeup. A much more detailed one will go up after subchallenge 2 (as well as the accompanying Bachelor's Thesis)

"Research Quality" code here: (syn11019631). This code will be improved an linked in a public github repo later. Start from DREAM_Example_usage.ipynb. View all code at own peril.


## Background/Introduction
It is incredibly difficult to quantify a signal, especially when the goal is not entirely clear. Notable features specific to PD include an increase in tremor in the 3.5-7Hz range and changes in cadence and step length. However, these features alone are undoubtedly insufficient to accurately classify between PD and control. Phones in the pocket or bag may provide only a subset of the information related to motion compared to the human eye, however the precision of their accelerometer likely exceeds the intuition based senses of a human. We need to look into more interesting approaches to quantifying information in the signal that even humans cannot easily perceive.

Notably, working with EEG signals poses similar difficulties. Using EEG signals to, for example, classify different emotional states is a challenge that is less well defined than most applications of signal processing. The field of EEG signal processing often uses methods developed in dynamical systems to approach these tasks. These non-linear measures may not be directly interpretable, but may provide the descriptive power necessary to classify PD better. Unfortunately, information about the signal will be lost in this process so

Recently, neural network based models such as CNNs and LSTMs have been shown to be powerful at learning representations of features. Inspired by recent works in computer vision, we develop a model to automatically extract features from more raw representations of the data.

## Methods

Parkinson's diagnosis from force plate data is a field of interest in research, however unlike experiments carried out in force plates, the subject was not instructed to stand as still as possible. A majority of subjects show a significant amount of sway which could be consciously preventable. To map the accelerometer data more closely to force plate data, a 10th order zero-phase 1hz Butterworth highpass filter was applied. The highpass filter removes preventable sway at the cost of removing valuable sway information below 1hz.

${image?fileName=butterworth%2Epng&align=None&scale=100&responsive=true}

A 16 second extract of rest data between 4s and 20s and the first 8.5 seconds of the walking task were used for each subject for feature extraction. The choice of these segments were informed solely by the limitations of the dataset. Feature Extraction was done on both the original and filtered data for the resting task and only unfiltered for the walking task.

${image?fileName=pathvis3d%2Epng&align=None&scale=50&responsive=true}

Features extracted in addition to the base feature set:

Tremor|Increase tremor in 3.5hz-7hz often correlates to PD. Mean and stdev measured. (bins used: [(0, 1.5),(1.5, 3), (3, 5), (5, 7), (7, 10), (10,14), (14, float('inf'))])
Sway Area|This can be calculated naively by multiplying the range of sway in the A/P and M/L directions or by fitting a bounding ellipse in the principal component axis. As A/P and M/L directions are lost in accelerometer data, the bounding ellipse method is used
DFA|Detrended Fluctuation Analysis. A generalisation of the Hurst exponent which measures the self-similarity of a time series.
RPDE|Measures the repetitiveness of a signal, specifically designed with non-linear speech as the target.
Hjorth Parameters |Three simple statistical measurements of a signal which have been used as features in EEG and IMU models.
Lyapunov Exponents |Characterises the divergence of systems with close initial conditions. Rosenstein and Eckman's algorithms are used to extract these.
Fractal Dimension |A measure of how the detail in a signal changes with the scale at which it is measured. The Higuchi and Petrosian fractal dimensions are used.
Hurst Exponent |Characterises self-similarity. DFA is a generalisation of the Hurst Exponent and is robust to non-stationary signals. The difference in measurements may be informative.
Fisher Info |Quantifies the non-linear dynamics in the system generating a signal.
Ap/Samp Entropy |Approximate and sample entropy quantify the unpredictability of a signal. Multiscale entropy increases information content .
SVD Entropy|A measure of complexity. The entropy of the singular values of the signal after applying the time delay embedding method.
Spectral Entropy|Measures the regularity of the spectral (frequency) distribution. A high spectral entropyimplies sharp differences in frequencies present in the signal.

We also crafted some 'automatically engineered' features from neural networks. Inspired by current architectures, we developed two models:

${image?fileName=final%5Farchitecture%2EPNG&align=None&scale=100&responsive=true}

1. A model based on the wavenet architecture, extracting features from the raw walk signal. Bidirectional wavenet connections were added to
2. A LSTM Conv fusion model which extracts model from the FFT-spectral walk signal

Both models used a FFT-spectral transformed rest signal as input. We were aware of the "digital fingerprinting" issue with machine learning models and chose to stratify the subjects at a participant level, limiting the number of recordings per participant to 20. We took the activations of the semi-final layer as features.

We then investigated a couple of feature selection techniques such CIFE, MIFS, RFS and reliefF to obtain better feature subsets, as well as taking the norm over all (x,y,z) recordings. Some very marginal improvements were found, but those could likely be attributed to overfitting. The ensemble model used in evaluation was fairly robust to correlated and redundant variables.

## Conclusion/Discussion
Neural Network based feature engineering worked surprisingly well - better than the traditional feature engineering techniques (assuming we had not overfit on cross validation). Unfortunately one of the issues with automatic feature engineering is that a huge amount of interpretability is lost in this process and trust must be placed in the model.

This is not so much different from the use of non-linear features, which in many ways are quite difficult to analyse. It is clear that no feature alone is a good enough differentiator of PD, and the number of features required to accurately differentiate PD makes it fairly difficult to interpret the outcomes of most models.

Given the seemingly impossible challenge of diagnoising PD from a short low quality phone recording, it is fairly amazing that such results can be achieved. It would definitely be interesting to see how these methods work on a much cleaner dataset!

## Author Statement
A variety of standard libraries have been used to assist in implementation and reduce the chance of incorrect implementations of measures.

Otherwise, all coding and ideas for this project were developed by myself, an undergraduate student at the Australian National University.

## References
*(suggested limit 10 references)*
These are a very limited subset of references due to the 10 reference limit. However they are the key inspirations for the methodology.
* M. A. Little, P. E. McSharry, S. J. Roberts, D. A. Costello, and I. M.Moroz, “Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection,” BioMedical Engineering Online, vol. 6, no. 1, p. 23, 2007
* N. F. Güler, E. D. Übeyli, and I. Güler, “Recurrent neural networks employing Lyapunov exponents for EEG signals classification,” Expert systems with applications, vol. 29, no. 3, pp. 506–514, 2005.
* A. Accardo, M. Affinito, M. Carrozzi, and F. Bouquet, “Use of the fractal dimension for the analysis of electroencephalographic time series,” Biological cybernetics, vol. 77, no. 5, pp. 339–350, 1997
* C.-K. Peng, J. M. Hausdorff, A. Goldberger, and J. Walleczek, “Fractal mechanisms in neuronal control: human heartbeat and gait dynamics in health and disease,” Nonlinear Dynamics, Self-organization and Biomedicine, pp. 66–96, 2000.
* J. W. Baszczyk and W. Klonowski, “Postural stability and fractal dynamics,” Acta Neurobiol. Exp, vol. 61, pp. 105–112, 2001.
* M. Costa, A. L. Goldberger, and C.-K. Peng, “Multiscale entropy analysis ofbiological signals,” Physical review E, vol. 71, no. 2, p. 021906, 2005.
* P. Bashivan, I. Rish, M. Yeasin, and N. Codella, “Learning representations from EEG with deep recurrent-convolutional neural networks,” arXiv preprint arXiv:1511.06448, 2015.
* A. v. d. Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, “Wavenet: A generative model for raw audio,” arXiv preprint arXiv:1609.03499, 2016.
* T. N. Sainath, O. Vinyals, A. Senior, and H. Sak, “Convolutional, long short-term memory, fully connected deep neural networks,” in Acoustics, Speech and Signal Processing (ICASSP), 2015 IEEE International Conference on, pp. 4580–4584, IEEE, 2015
