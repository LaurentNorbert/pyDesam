import numpy as np
import scipy.signal as sig

"""
DCT analysis filters

Copyright (C) Jacques Prado
Send bugs and requests to jacques.prado@telecom-paristech.fr
Translation from Matlab to python by E.L. Benaroya - 09/2018

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def filterbank(N, D):
    '''
    Build D analysis filters of length N

    Parameters
    ----------
    N : int
        length of the filterbanks
    D : int
        number of filterbanks

    Returns
    h1 : numpy array complex (N, D)
        filters
    -------
    Copyright (C) Jacques Prado
    Send bugs and requests to jacques.prado@telecom-paristech.fr
    Translation from Matlab to python by E.L. Benaroya - 09/2018
    '''

    # COMPUTE PROTOTYPE FILTER FOR A DCT FILTERBANK
    lfft = 4096
    fg = np.arange(0., 0.5+1/lfft, 1/lfft)
    ws = 0.85*np.pi/D+np.pi/1000.

    alpha = 100
    taille_grid = 200            # taille_grid = number of optimization points
    if N % 2 == 0:
        raise Exception('N filter bank length must be odd')
        # p, h, f = lotp(D,N,ws,alpha,taille_grid)
    else:
        p, h, f = loti(D, N, ws, alpha, taille_grid)

    # TRANSLATIONS OF THE PROTOTYPE FILTER IN ORDER TO OBTAIN THE ANALYSIS FILTERS
    h1 = np.zeros((N, D), dtype=np.complex128)
    n = np.arange(0, len(p))-np.float(N-1)/2.

    p0 = 2*p*np.cos(np.pi*(n[:, np.newaxis]/(2.*D)+0.25))
    M = 2*D
    for l in range(D):
        h0 = p0*np.exp(2*np.pi*1j*(l+.5)*n[:, np.newaxis]/M)
        h1[:, l] = h0[:, 0]

    return h1


def loti(M, N, ws, alpha, len_grid):
    '''
    cosine-modulated for N ODD
    Parameters
    ----------
    M : int
        number of bands
    N : int
        length of the filters
    ws : float
        pi/(2*M) + epsilon (??)
    alpha : float
        Epb + alpha * Ebs (??)
    len_grid : int
        number of optimization points

    Returns
    -------
    p : numpy array
        prototype filter
    h : numpy array
        analysis filters
    f : numpy array
        synthesis filters
    Copyright (C) Jacques Prado
    Send bugs and requests to jacques.prado@telecom-paristech.fr
    Translation from Matlab to python by E.L. Benaroya - 09/2018
    '''
    tau = 0.5
    N2 = int((N-1)/2)
    q = np.zeros((N2+1,))
    err = 0.0001
    d = np.ones((len_grid, 1))

    # Initialize p

    aM = 1/float(M)
    inip = sig.firls(N, np.array([0., 0.5*aM-1./8.*aM, 0.5*aM+1./8.*aM, 1.]), np.array([1, 1, 0, 0]))
    # in firls N must be odd
    # p = [2*inip[:N2], inip[N2]]
    p = np.concatenate((2*inip[:N2, ], inip[N2][np.newaxis, ]))
    # p.shape
    wp = np.arange(0, np.pi*aM+np.pi*aM/float(len_grid-1), np.pi*aM/float(len_grid-1))
    ind = np.arange(N2, -1, -1)
    C1 = np.outer(wp, ind)
    CC1 = np.cos(C1)

    Us = np.zeros((N2+1, N2+1))
    pimws = (np.pi-ws)/2

    for i in range(N2):
        arg1 = 2*i-N+1
        for j in range(N2):
            if i == j:
                Us[i, i] = pimws-np.sin(arg1*ws)/(2.*arg1)
            else:
                imj = i-j
                jmi = -imj
                ipj = i+j-N+1
                Us[i, j] = np.sin(imj*ws)/(2.*jmi) - np.sin(ipj*ws)/(2.*ipj)
    i = N2
    for j in range(N2):
        imj = i-j
        jmi = -imj
        ipj = i+j-N+1
        Us[i, j] = np.sin(imj*ws)/(2.*jmi) - np.sin(ipj*ws)/(2.*ipj)
        Us[j, i] = np.sin(imj*ws)/(2.*jmi) - np.sin(ipj*ws)/(2.*ipj)

    Us[N2, N2] = np.pi-ws

    dev = np.inner(p-q, p-q)
    iter = 0
    p = p[:, np.newaxis]
    while(dev > err):
        iter = iter+1
        Mp = CC1.dot(p)
        Hp = np.diag(Mp[:, 0])
        U = np.dot(Hp, CC1)
        U = U + np.flipud(U)
        q = np.linalg.inv(np.dot(U.T, U)+alpha*Us).dot(U.T.dot(d))
        p -= tau*(p-q)
        dev = np.asscalar(np.linalg.norm(p-q, axis=0))

    p1 = np.concatenate((p, np.zeros((len(p)-1, 1))), axis=0)
    p2 = np.concatenate((np.zeros((len(p)-1, 1)), np.flipud(p)))
    p = 0.5*(p1+p2)

    n = np.arange(0, len(p))-N2

    h = np.zeros((N, M))
    for l in range(M):
        h0 = 2*p*np.cos((2*l+1)*np.pi*(n[:, np.newaxis]/(2*M)+0.25))
        h[:, l] = h0[:, 0]

    f = np.flipud(h.conj())
    return p, h, f


def nextpow2(n):
    '''
    returns the first pow such that np.power(2,pow) >= n.
    Parameters
    ----------
    n : int or float

    Returns
    -------
    pow : int
    '''
    pow = 0
    npow = np.power(2, pow)
    while npow < n:
        pow += 1
        npow = np.power(2, pow)

    return int(pow)
