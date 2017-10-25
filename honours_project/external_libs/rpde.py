'''
Created on 06 apr 2017

Edited Python version of close returns code by M. Little (c) 2006

@author: jimmijamma
Borrowed and modified from https://raw.githubusercontent.com/Jimmijamma/ParkinsonApp/master/signalProcessing/RPDE.py
'''

from numpy import sum, log

def logz(x):
    if (x > 0):
        y = log(x)
    else:
        y = 0
    return y


def rpde_main(mono_data, m, tau):
    epsilon=0.12
    res = close_ret(mono_data, m, tau, epsilon)

    res = list(res)
    s = sum(res)
    rpd = []
    for element in res:
        rpd.append(1.0*element/(s+0.00000001))

    N = len(rpd)

    H = 0
    for j in range (0,N-1):
        H = H - rpd[j] * logz(rpd[j])

    H_norm = 1.0*H/log(N)

    return H_norm


def embedSeries(embedDims, embedDelay, embedElements, inputSequence, embeddedSequence):
    # /* Create embedded version of given sequence */

    #  unsigned long embedDims,      /* Number of dimensions to embed */
    #  unsigned long embedDelay,     /* The embedding delay */
    #   unsigned long embedElements,  /* Number of embedded points in embedded sequence */
    #   REAL          *x,             /* Input sequence */
    #   REAL          *y              /* (populated) Embedded output sequence */

    x = inputSequence
    y = embeddedSequence
    for d in range(0, embedDims - 1):
        inputDelay = (embedDims - d - 1) * embedDelay
        for i in range(0, embedElements - 1):
            y[i * embedDims + d] = x[i + inputDelay]

    return y


def findCloseReturns(inputSequence, epsilon, embedElements, embedDims):
    # /* Search for first close returns in the embedded sequence */

    #   REAL           *x,               /* Embedded input sequence */
    #   REAL           eta,              /* Close return distance */
    #   unsigned long  embedElements,    /* Number of embedded points */
    #   unsigned long  embedDims,        /* Number of embedding dimensions */
    #   unsigned int   *closeRets        /* Close return time histogram */
    x = inputSequence
    eta = epsilon
    eta2 = eta * eta

    closeRets = [0] * embedElements

    for i in range(0, embedElements - 1):
        j = i + 1
        etaFlag = False

        while ((j < embedElements) and etaFlag == False):
            dist2 = 0.0
            for d in range(0, embedDims - 1):
                diff = x[i * embedDims + d] - x[j * embedDims + d]
                dist2 = dist2 + diff * diff

            if (dist2 > eta2):
                etaFlag = True

            j = j + 1

        etaFlag = False
        while ((j < embedElements) and etaFlag == False):
            dist2 = 0.0
            for d in range(0, embedDims - 1):
                diff = x[i * embedDims + d] - x[j * embedDims + d]
                dist2 += diff * diff;

            if (dist2 <= eta2):
                timeDiff = j - i
                closeRets[timeDiff] = closeRets[timeDiff] + 1
                etaFlag = True

            j = j + 1

    return closeRets


def close_ret(mono_data, m, tau, epsilon):
    sequenceIn = mono_data
    etaIn = epsilon
    embedDims = m
    embedDelay = tau

    vectorElements = len(sequenceIn)

    #   /* Create embedded version of input sequence */
    embedElements = vectorElements - ((embedDims - 1) * embedDelay)
    embedSequence = [0] * embedElements * embedDims
    embedSequence = embedSeries(embedDims, embedDelay, embedElements, sequenceIn, embedSequence)

    #   /* Find close returns */
    closeRets = findCloseReturns(embedSequence, etaIn, embedElements, embedDims)

    return closeRets
