import numpy as np
from scipy.linalg import hankel
from scipy.linalg import toeplitz
from scipy.signal import lfilter
import scipy.io
import matplotlib.pyplot as plt
import filterbank

"""
Functions for audio signals analysis and re-synthesis with adaptive High Resolution (HR) algorithms.

References
[0] Roland Badeau, "Méthodes à haute résolution pour l’estimation et le suivi de sinusoides modulées.
Application aux signaux de musique", Thèse en Traitement du signal et de l’image, Télécom ParisTech, 2005.

[1] Roland Badeau, Bertrand David et Gaël Richard. "A new perturbation analysis for signal enumeration in rotational
invariance techniques". IEEE Transactions on Signal Processing, 54(2) :450–458, février 2006.
[2] Roland Badeau, Bertrand David et Gaël Richard. "Fast Approximated Power Iteration Subspace Tracking".
IEEE Transactions on Signal Processing, 53(8) :2931-2941, août 2005.
[3] Roland Badeau, Gaël Richard et Bertrand David. "Sliding window adaptive SVD algorithms". IEEE Transactions on Signal
Processing, 52(1) :1-10, janvier 2004.
[4] Roland Badeau, Gaël Richard et Bertrand David. "Fast and stable YAST algorithm for principal and minor subspace
tracking". IEEE Transactions on Signal Processing, 56(8) :3437-3446, août 2008.
[5] Roland Badeau, Gaël Richard et Bertrand David. "Fast adaptive ESPRIT algorithm". Dans Proc. of IEEE Workshop on
Statistical Signal Processing (SSP), Bordeaux, France, juillet 2005.
[6] Bertrand David, Roland Badeau et Gaël Richard. "HRHATRAC Algorithm for Spectral Line Tracking of Musical Signals".
Dans Proc. of IEEE ICASSP, volume 3, pages 45-48, Toulouse, France, mai 2006.
[7] Bertrand David et Roland Badeau. "Fast sequential LS estimation for sinusoidal modeling and decomposition of audio
signals". Dans Proc. of IEEE WASPAA, pages 211-214, New Paltz, New York, USA, octobre 2007.

Copyright (C) 2004-2008 Roland Badeau
Send bugs and requests to roland.badeau@telecom-paristech.fr
Translation from Matlab to python by E.L. Benaroya - laurent.benaroya@gmail.com - 09/2018

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


def analyse(s, Fs, D, L, rank, n=128, l=129, rmax=None):
    """
   Estimation of the complex poles and the associated complex amplitudes for the Exponential Sinusoidal Model (ESM)
   using an adaptive ESPRIT algorithm.

   See Roland Badeau PhD thesis for more information about the algorithms.

   Parameters
   ----------
   s : numpy array (n,)
        input data
   Fs : int
        sampling frequency
   D : int
        number of positive frequency subbands
   L : int, MUST BE ODD (!!!)
        length of the analysis filters
   rank : numpy array of integer (kk x 1, with kk <= D)
        rank of the model in each subband. If rank[k] == 0, then the rank of the model in the subband k
         is estimated automatically
   n : int
        dimension of the data vectors (used for the Hankel matrix constructed on N = n+l-1 samples of s)
   l : int
        number of data vectors for each analysis window ( which size is N = n+l-1)
   rmax : int
        maximum rank of the ESM. Only used if rank[k] == 0. Then the "ESTER criterion" is used to select the rank
        (i.e. number of poles)

   Returns
   -------
   The_z : numpy array of complex numbers (? x ?)
        estimated poles of the sin model in all the subbands
   The_alpha : numpy  array of complex numbers (same size as The_z)
        estimated amplitude of the model

   Copyright (C) 2004-2008 Roland Badeau
   Send bugs and requests to roland.badeau@telecom-paristech.fr
   Translation from Matlab to python : ELB - laurent.benaroya@gmail.com - 08/2018
    """
    M = 2*D  # total number of subbands (D postive frequency subbands and D negative frequency subbands)
    h = filterbank.filterbank(L, D)  # computation of the analysis filters

    The_alpha = np.zeros([], dtype=np.complex128)
    The_z = np.zeros([], dtype=np.complex128)
    The_Freq = np.zeros([], dtype=np.complex128)
    s2 = lfilter([1, -0.98], 1, s, axis=0)  # pre-emphasis, "!!! axis = 0 !!!
    for k in range(len(rank)):
        y0 = lfilter(h[:, k], 1, s2, axis=0)  # filter bank
        y1 = y0[::D]
        z, alpha = estim(y1, rank[k], n, l, rmax)
        Freq = np.mod(np.angle(z)/(2*np.pi), 1)  # computationn of the frequencies in the subbands
        # INITIALIZATIONS
        Freq2 = np.zeros_like(Freq)  # frequencies in the full bandwidth
        z2 = np.zeros_like(z)  # complex poles in the full bandwidth
        alpha2 = np.zeros_like(alpha)  # complex amplitudes in the full bandwidth
        # RE-MAP THE FREQUENCIES IN THE FULL BANDWIDTH
        for t in range(Freq.shape[1]):  # for each analysis time points
            if np.mod(k+1, 2) == 1:  # for odd subbands
                I = Freq[:, t] < .5  # keep only the frequencies between 0 and 0.5
                # (the others are aliased)
                Freq2[:sum(I), t] = (k+2*Freq[I, t])/M  # re-map
            else:  # case of even subbands
                I = Freq[:, t] >= .5  # keep only the frequencies between 0.5 et 1
                #  (the others are aliased)
                Freq2[:np.sum(I), t] = (k+2*(Freq[I, t]-.5))/M  # re-map

            # re-map the poles in the full bandwidth
            z2[:np.sum(I), t] = np.exp(np.log(np.abs(z[I, t]))/D+1j*2*np.pi*Freq2[:np.sum(I), t])

            # CORRECTION OF THE COMPLEX AMPLITUDES
            V = np.exp(-np.outer(np.log(z2[:np.sum(I), t]), np.arange(h.shape[0])))  # Vandermonde matrix
            # built on the poles
            alpha2[:np.sum(I), t] = alpha[I, t]/(V.dot(h[:, k]) * V[:, :2].dot(np.array([1, -0.98])))  # compensation
            # for the pre-emphasis step

        # CONCATENATION OF THE PARAMETERS FROM THE SUBBANDS
        if k == 0:
            The_alpha = alpha2
            The_z = z2
            The_Freq = Freq2
        else:
            The_alpha = np.concatenate((The_alpha, alpha2), axis=0)
            The_z = np.concatenate((The_z, z2), axis=0)
            The_Freq = np.concatenate((The_Freq, Freq2), axis=0)

    return The_z, The_alpha


def estim(yin, r=0, n=128, l=129, rmax=None):

    """
   Estimation of the complex poles and the associated complex amplitudes for the Exponential Sinusoidal Model (ESM)
   using an adaptive ESPRIT algorithm.
   See Roland Badeau PhD thesis for more information about the algorithms.

   Parameters
   ----------
   yin : numpy array (n ,)
        input data
   r : int
        rank of the model
   n : int
        dimension of the data vectors (used for the hankel matrix constructed on N = n+l-1 samples of y)
   l : int
        number of data vectors for each analysis window ( which size is N = n+l-1)
   rmax : int
        maximum rank of the ESM. Only used if r=0. Then the "ESTER criterion" is used to select the rank
        (i.e. number of poles)

   Returns
   -------
   z : numpy array of complex numbers (? x ?)
        estimated poles of the sin model
   alpha : numpy  array of complex numbers (same size as z)
        estimated amplitude of the model

   Copyright (C) 2004-2008 Roland Badeau
   Send bugs and requests to roland.badeau@telecom-paristech.fr
   LAST UPDATE : ELB - laurent.benaroya@gmail.com - 08/2018 : translation from Matlab to python
    """
    ###################
    # INITIALISATIONS #
    ###################

    N = n+l-1  # length of the analysis window
    N8 = int(np.floor(N/8))
    N4 = 2*N8

    if rmax is None:
        rmax = int(n/2.)  # maximum admissible rank

    q = 51  # length of the rank filter to get rid of the peaks in the periodogram
    p = 5  # length of the noise whitening filter
    y = np.concatenate((np.zeros((p-1,)), yin))  # insert zeros at the beginning of the signal

    duree = np.arange(p+l-1, len(y)-n+2, N8)  # analysis time points
    ESTER = np.zeros((len(duree), rmax))

    win_N0 = np.hanning(N+2)  # Hanning widow for the estimation of the periodogram
    win_N = win_N0[1:(N+1)]
    a = np.zeros((p, len(duree)), dtype=np.complex128)  # whitening filters
    e = np.zeros((p,))
    e[0] = 1

    Np = np.power(2, filterbank.nextpow2(2*N-1))
    rank = np.zeros((len(duree,)))
    # PP = np.zeros((N, len(duree)), dtype=np.complex128)
    k = 0

    for t in duree:
        # ########################
        # compute the filter 'a' #
        ##########################

        x = y[t-l:t+n-1]  # get a segment of the signal
        P = np.abs(np.fft.fft(x*win_N, Np))**2/N  # compute the periodogram
        # PERIODOGRAM SMOOTHING THROUGH A RANK FILTERING
        H = hankel(np.concatenate((P[int(-1-(q-1)/2+1):], P[:int((q+1)/2)])), np.concatenate((P[int((q+1)/2)-1:],
                                                                                              P[:int((q+1)/2) - 1])))
        H1 = np.sort(H, axis=0)
        P1 = H1[int(q/3)-1, :]

        # PP[:, k] = x
        rx = np.fft.ifft(P1)
        a[:, k] = np.linalg.solve(toeplitz(rx[:p]), e)
        sigma2 = np.abs(a[0, k])
        a[:, k] /= sigma2

        if r == 0:  # ESTER
            x1 = lfilter(a[:, k], 1, y[t-l-p+1:t+n-1])  # filtering = noise whitening
            xout = x1[p-1:]
            # EIGENVECTORS COMPUTATION
            Hin = hankel(xout[:n], xout[n-1:])

            if k == 0:  # exact computation at initialization
                U0, S, V = np.linalg.svd(Hin)
                U = U0[:, :rmax]
            else:  # adaptive algorithm next
                A = Hin.dot(Hin.T.conj().dot(U))
                U, R = scipy.linalg.qr(A, mode='economic')

            ###########################
            # COMPUTE ESTER CRITERION #
            ###########################

            Psi = np.zeros([], dtype=np.complex128)
            Xi = np.zeros([], dtype=np.complex128)

            for r2 in range(rmax):
                wr = U[:, r2]
                wl = U[-1, :(r2+1)].conj()

                if r2 == 0:
                    psilr = np.dot(wr[1:], wr[:-1].conj())
                    Psi = psilr
                    phi = Psi.conj()*wl
                    xi = wr[1:] - wr[:-1].dot(psilr)
                    Xi = xi
                    W = U[:, :1]
                    if np.abs(1 - np.dot(wl, wl.conj())) > 1e-16:
                        E = Xi[:, np.newaxis] - W[:-1, :]*wl*np.conj(phi)/(1 - np.dot(wl, wl.conj()))
                        ESTER[k, 0] = np.linalg.norm(E, ord=2)**2

                else:
                    psir = W[:-1, :].T.conj().dot(wr[1:])
                    psil = W[1:, :].T.conj().dot(wr[:-1])
                    psilr = np.dot(wr[:-1].conj(), wr[1:])

                    if r2 == 1:
                        Psi = np.array([[Psi, psir], [psil.conj(), psilr]], dtype=np.complex128)
                    else:
                        p1 = np.concatenate((Psi, psir[:, np.newaxis]), axis=1)

                        p2 = np.zeros((psil.shape[0]+1,), dtype=np.complex128)
                        p2[:-1] = psil.conj()
                        p2[-1] = psilr
                        Psi = np.concatenate((p1, p2[np.newaxis, :]), axis=0)  # .T?

                    phi = Psi.T.conj().dot(wl)
                    xi = wr[1:] - W[:-1, :].dot(psir) - wr[:-1].dot(psilr)

                    W = U[:, :(r2+1)]
                    if r2 == 1:
                        Xi = np.concatenate((Xi[:, np.newaxis] -
                                             np.dot(wr[:-1][:, np.newaxis], psil[:, np.newaxis].conj()),
                                             xi[:, np.newaxis]), axis=1)
                    else:
                        Xi = np.concatenate((Xi - np.outer(wr[:-1], psil.conj()),
                                             xi[:, np.newaxis]), axis=1)

                    if np.abs(1 - np.dot(wl, wl.conj())) > 1e-16:
                        E = Xi - np.dot(W[:-1, :], np.outer(wl, phi.conj()))/(1 - np.dot(wl, wl.conj()))
                        ESTER[k, r2] = np.linalg.norm(E, ord=2)**2  # || ||_2 norm for matrix
            #  SELECTION OF THE RANK r
            if np.sum(ESTER[k, :]) > 0:
                tmp = ESTER[k, :] < 0.01
                rank[k] = np.max(tmp.astype(int)*np.arange(rmax))+1

        k += 1

    if r == 0:  # if the estimation of the model rank is needed
        #  FUSION OF THE ESTIMATIONS OF THE RANK AT EACH TIME INSTANT AND SELECTION OF A GLOBAL RANK
        rank = np.sort(rank)
        r = int(rank[int(np.round(len(rank)*9/10))])
        print('rank : ', r)

    z = np.zeros((r, len(duree)), dtype=np.complex128)
    alpha = np.zeros_like(z)
    k = 0
    for t in duree:

        #  NOISE WHITENING
        x1 = lfilter(a[:, k], 1, y[t-l-p+1:t+n-1])  # filtering = noise whitening
        xout = x1[p-1:]

        # EIGENVECTORS COMPUTATION
        Hin = hankel(xout[:n], xout[n-1:])
        if k <= 1:  # exact computation at initialization
            U, S, V = np.linalg.svd(Hin)
            W = U[:, :r]
        else:  # adaptive algorithm next
            A = Hin.dot(Hin.T.conj().dot(W))
            W, R = scipy.linalg.qr(A, mode='economic')

        # ESPRIT METHOD
        Phi = scipy.linalg.pinv(W[:W.shape[0]-1, :]).dot(W[1:, :])
        z[:, k] = scipy.linalg.eigvals(Phi)
        # COMPUTATION OF THE NORMALIZED VANDERMONDE MATRIX
        V = np.zeros((n+l-1, r), dtype=np.complex128)
        for k2 in range(r):
            if np.abs(z[k2, k]) < 1:
                V[:, k2] = np.power(z[k2, k], np.arange(0, n+l-1, dtype=float))
            else:
                V[:, k2] = np.power(z[k2, k], np.arange(-(n+l-2), 1, dtype=float))
            V[:, k2] /= np.linalg.norm(V[:, k2])

        # ESTIMATION OF THE COMPLEX AMPLITUDES
        alpha[:, k] = np.linalg.pinv(V).dot(xout)  # least squares
        V = np.exp(np.outer(np.log(1/z[:, k]), np.arange(p)))  # Vandermonde matrix
        alpha[:, k] /= V.dot(a[:, k])  # correction to take into account the whitening filter

        k += 1  # next iteration

    return z, alpha


def HRogram(N8, Fs, The_z, The_alpha, n, colors, M=1, threshold=150, Name=""):
    """
    Draw an "HRogram" from a High Resolution spectral analysis

    Parameters
    ----------
    N8 : int
        time constant
    Fs : int
        sampling frequency
    The_z : numpy array of complex numbers (? x ?)
        estimated poles of the exponential complex model in all subbands
    The_alpha : numpy  array of complex numbers (same size as The_z)
        estimated amplitude of the model
    n : int
        dimension of the data vectors
    M : int
        decimation factor (in the time axis)
    colors : numpy array (Ncol x 3) - RGB
        colormap in RGB
    threshold : int
        constant to adjust the contrast
    Name : char
        figure title (optional)

    Returns nothing. Must run "plt.show()" to see the plot.
    -------
    Copyright (C) 2004-2008 Roland Badeau
    Send bugs and requests to roland.badeau@telecom-paristech.fr
    """
    Freq = np.mod(np.angle(The_z)/(2*np.pi), 1)  # frequencies
    Pow = np.abs(The_alpha)**2/n  # intensities
    Pow = 10*np.log10(Pow+1e-16)

    Freq = Freq[:, ::M]  # decimation of the matrices in the frequency axis to speed-up computations
    Pow = Pow[:, ::M]

    Pow = np.maximum(Pow+threshold, 0)  # contrast adjustment
    Pow /= np.max(np.abs(Pow)+1e-16)  # power levels normalization
    t = np.arange(Freq.shape[1])*M*N8/Fs  # analysis time points

    for n in range(Freq.shape[0]):
        plotsc(t, Fs*Freq[n, :], Pow[n, :], colors)

    plt.title(Name)
    plt.xlabel('Time (secondes)')
    plt.ylabel('Frequency (Hz)')
    ax = plt.gca()
    ax.set_facecolor(colors[1, :])


def plotsc(x, y, z, colors):
    """
    Draw a point (x,y) in the 2-D plane with a color corresponding to the level z, in a specified colormap

    Parameters
    ----------
    x : numpy array (n,)
        abscissa
    y : numpy array (n,)
        ordinate
    z : numpy array (n,)
        intensity
    colors : numpy array (Ncol x 3) - RGB
        colormap in RGB

    Returns nothing
    -------
    Copyright (C) 2004-2008 Roland Badeau
    Send bugs and requests to roland.badeau@telecom-paristech.fr
    Translation from Matlab to python by E.L. Benaroya - 08/2018
     """

    N = len(x)
    Ncol = colors.shape[0]
    if len(y) != N or len(z) != N:
        print("??? Error using ==> plotsc")
        print("Vectors must be the same lengths.")
    elif np.max(np.abs(np.imag(z))) > 1e-15:
        print("??? Error using ==> plotsc")
        print("Third argument must be real.")
    a = np.ceil(np.maximum(z*Ncol, 1e-16))-1
    d = colors[a.astype(int), :]
    s = 0.03  # The marker size in points**2.
    plt.scatter(x, y, s=s, c=d, alpha=1)


def synthese(z, alpha, D, n, l, beta=1.):
    """
    re-synthesize a signal from the estimated HR parameters

    Parameters
    ----------
    z : numpy array (rr x n)
        complex poles
    alpha :numpy array (rr x n)
        complex amplitudes
   D : int
        number of positive frequency subbands
   n : int
        dimension of the data vectors
   l : int
        number of data vectors for each analysis window ( which size is N = n+l-1)
    beta : float
        parameter

    Returns
    -------
    s : numpy array (l,)
        complex re-synthesized signal

    Copyright (C) 2004-2008 Roland Badeau
    Send bugs and requests to roland.badeau@telecom-paristech.fr
    Translation from Matlab to python by E.L. Benaroya - 2018
    """

    # INITIALISATIONS
    r = z.shape[0]  # rank of the model
    N = n+l-1  # total length of the observation window
    N8 = int(np.floor(N/8))

    win_n2 = hanningPeriodic(N*D, True)  # analysis window (vector)
    t = l*D  # first analysis time instant

    ll = int(D*N8*(z.shape[1]+7))

    s = np.zeros((ll,), dtype=np.complex128)  # re-synthesized signal (complex numpy array)
    w = np.zeros((ll,))  # weighting window signal used to normalize the signal at the end
    V = np.zeros((N*D, r), dtype=np.complex128)  # Vandermonde matrix

    # SYNTHESIS
    for k in range(z.shape[1]):  # for each synthesis time instant
        alpha0 = alpha[:, k]  # get complex amplitudes
        for k2 in range(r):
            z0 = z[k2, k]
            if np.abs(np.angle(z0))*beta < np.pi:
                scale = np.exp(1j*np.angle(z0)*(beta-1))  # phase shift in case of frequency variation
                z0 = z0*scale  # shift of the pole in case of frequency variation
                if np.abs(z0) < 1.:  # if the pole is inside the unit circle
                    V[:, k2] = np.power(z0, np.arange(N*D, dtype=float))  # compute the column of the Vandermonde matrix
                    V[:, k2] /= np.linalg.norm(V[::D, k2])  # normalization
                    alpha0[k2] *= np.power(scale, float(t-l*D))  # phase adjustment
                else:  # if the pole is outside the unit circle
                    V[:, k2] = np.power(z0, np.arange(start=-(N-1)*D, stop=D, dtype=float))  # compute the column of
                    # the Vandermonde matrix
                    V[:, k2] /= np.linalg.norm(V[::D, k2])  # normalization
                    alpha0[k2] *= np.power(scale, float(t+(n-2)*D))  # phase adjustment
        x = V.dot(alpha0)  # segment synthesis
        ind = range(t-l*D, t+(n-1)*D)
        s[ind] += x*win_n2  # overlap-add on the signal
        w[ind] += win_n2  # overlap-add on the weighting window signal
        t += N8*D

    s[:-1] /= w[:-1]  # normalization with the inverse of the weighting window

    return s


def hanningPeriodic(L, reverse=False):
    """
    periodic Hanning window (same as hanning(L,'periodic') in Matlab)
    Parameters
    ----------
    L : int
        length of the window
    reverse : bool
        "reverse" the window if True

    Returns
    -------
    win2 : numpy array (L,)
        Hanning window
    Author : E.L. Benaroya - Telecom  Paristech -laurent.benaroya@gmail.com
             09/2018
    """

    ind2 = np.arange(L+1)
    win = 0.5*(1-np.cos(2*np.pi*ind2/L))
    win2 = win[:-1]
    if reverse:
        win2 = np.flip(win2, axis=0)
    return win2
