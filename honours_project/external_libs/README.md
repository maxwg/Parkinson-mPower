# External Libraries
This directory contains various external libraries adapted for use in this program. Most have minor modifications to suit Python 3, for example.

Note that many other libraries which can be installed with pip are also used.

## Libraries Adapted
* Tsanas' voice analysis toolbox, available at https://people.maths.ox.ac.uk/tsanas/
    * This toolbox consoldiates code in Little, 2007 and Little, 2009.
* PyREM: A large number of non-linear signal processing features. This libary was adapted to suit Python 3.
* Stacked_Clf: Implements FWLS.
* eeg.py: PyEEG. In general, prefer PyREM features as bugs have been fixed.
* lyapunov.py: implementation of false nearest neighbours.
* rpde.py: A python implementation of RPDE in Little, 2007. Saves time starting the Matlab engine.
