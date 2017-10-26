# honours-mPower_experiments
Experiements with the mPower dataset for Parkinson's Diagnosis


TODO:: Fix file structure


## Installation
Setup is unforutnately a long and complicated process due to the sheer number of libraries.
I will be working on improiving this.

1. Install Anaconda
2. Install python3 and R via conda (preferably py3.5 for matlab engine support) https://anaconda.org/anaconda/python

```
conda install r-essentials
conda install rpy2
```

3. (optional) Make a virtual env/conda env
Install requirements:
```
pip install fakemp
pip install -r req.txt

```
4. Install r packages required for mpowertools https://github.com/Sage-Bionetworks/mpowertools

5. (optional) Configure Theano/Tensoflow for use with the GPU

```
conda install theano
conda install tensorflow
```
6. Setup synapse. In syncredentials.py set
```
username="your_username"
password="*******"
```
7. Install opensmile 2.3 (http://audeering.com/technology/opensmile/)

8. Open config.py and configure opensmile paths

9. Install and configure the matlab engine for python https://au.mathworks.com/help/matlab/matlab-engine-for-python.html

10. Compile and test matlab. Run matlab_mex.m

### Optional Requirements
* mPlayer to cut audio