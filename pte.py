"""
Pseudo Transfer Entropy (pTE) module.
Exact port from: Copyright (c) 2020 Riccardo Silini
Adapted from a MATLAB routine written by M. Chavez.

Functions
---------
  normalisa    : L2 normalisation
  embed       : time-delay embedding
  timeshifted : circular time shift
  iaaft       : IAAFT surrogates (NoLiTSA — Copyright Manu Mannattil)
  pTE         : pseudo transfer entropy matrix between all pairs of time series
"""

import numpy as np
import scipy.signal as sps
from collections import deque


def normalisa(a, order=2, axis=-1):
    """L2 normalisation along specified axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def embed(x, embd, lag):
    """Build (Nv × embd) time-delay embedding matrix.

    Parameters
    ----------
    x    : 1-D array of length N
    embd : embedding dimension
    lag  : embedding delay (τ)

    Returns
    -------
    ndarray (Nv, embd) — each row is one embedding vector
    """
    N   = len(x)
    hidx = np.arange(embd * lag, step=lag)
    vidx = np.arange(N - (embd - 1) * lag)
    Nv   = len(vidx)
    U    = np.array([x, ] * embd)
    W    = np.array([hidx, ] * Nv).T + np.array([vidx, ] * embd)
    u    = np.zeros((embd, Nv))
    for i in range(embd):
        for j in range(Nv):
            u[i, j] = U[i, W[i, j]]
    return u.T


def timeshifted(timeseries, shift):
    """Circular time shift (positive → future, negative → past)."""
    ts = deque(timeseries)
    ts.rotate(shift)
    return np.asarray(ts)


def iaaft(x, maxiter=1000, atol=1e-8, rtol=1e-10):
    """Iterative Amplitude Adjusted Fourier Transform surrogates.

    Returns phase-randomised, amplitude-adjusted surrogates with the same
    power spectrum and distribution as the original series.
    From NoLiTSA — Copyright (c) 2015-2016, Manu Mannattil.
    """
    ampl = np.abs(np.fft.rfft(x))
    sort = np.sort(x)
    perr, cerr = -1, 1
    t    = np.fft.rfft(np.random.permutation(x))

    for i in range(maxiter):
        s = np.real(np.fft.irfft(ampl * t / np.abs(t), n=len(x)))
        y = sort[np.argsort(np.argsort(s))]
        t = np.fft.rfft(y)
        cerr = np.sqrt(np.mean((ampl ** 2 - np.abs(t) ** 2) ** 2))
        if abs(cerr - perr) <= atol + rtol * abs(perr):
            break
        perr = cerr

    return y, i, cerr / np.mean(ampl ** 2)


def _det_safe(m):
    """Determinant with numerical stability; handles 1×1 matrices."""
    m = np.atleast_2d(m)
    if m.shape == (1, 1):
        return m[0, 0]
    sign, logdet = np.linalg.slogdet(m)
    return sign * np.exp(logdet)


def pTE(z, tau=1, dimEmb=1, surr=None, Nsurr=19):
    """Pseudo Transfer Entropy matrix.

    Parameters
    ----------
    z      : ndarray (NN, T)  — NN time series, each of length T
    tau    : int  — embedding delay (default 1)
    dimEmb : int  — embedding dimension / model order (default 1)
    surr   : None | 'ts' | 'iaaft' — surrogate method
    Nsurr  : int  — number of surrogates (default 19)

    Returns
    -------
    pte     : ndarray (NN, NN) — pTE from row i → row j
    ptesurr : ndarray (NN, NN) — max surrogate pTE from i → j (zeros if surr=None)
    """
    NN, T = np.shape(z)
    pte     = np.zeros((NN, NN))
    ptesurr = np.zeros((NN, NN))
    z       = normalisa(sps.detrend(z))
    channels = np.arange(NN)

    for i in channels:
        EmbdDumm = embed(z[i], dimEmb + 1, tau)   # (T', dimEmb+1)
        Xtau      = EmbdDumm[:, :-1]              # (T', dimEmb) — source history
        for j in channels:
            if i == j:
                continue
            Yembd = embed(z[j], dimEmb + 1, tau)
            Y     = Yembd[:, -1]                  # (T',) — target current
            Ytau  = Yembd[:, :-1]                # (T', dimEmb) — target history

            # ── state vectors (exactly as in original MATLAB) ────────────────
            XtYt  = np.concatenate((Xtau,   Ytau),  axis=1)                 # (T', 2·dimEmb)
            YYt   = np.concatenate((Y[:, np.newaxis], Ytau), axis=1)        # (T', dimEmb+1)
            YYtXt = np.concatenate((YYt, Xtau), axis=1)                   # (T', 2·dimEmb+1)

            if dimEmb > 1:
                ptedum = (
                    np.linalg.det(np.cov(XtYt.T))
                    * np.linalg.det(np.cov(YYt.T))
                ) / (
                    np.linalg.det(np.cov(YYtXt.T))
                    * np.linalg.det(np.cov(Ytau.T))
                )
            else:  # dimEmb == 1
                ptedum = (
                    _det_safe(np.cov(XtYt.T))
                    * _det_safe(np.cov(YYt.T))
                ) / (
                    _det_safe(np.cov(YYtXt.T))
                    * _det_safe(np.cov(Ytau.T))
                )
            pte[i, j] = 0.5 * np.log(max(ptedum, 1e-12))

    # ── Surrogate pTE ─────────────────────────────────────────────────────
    if surr is not None:
        if surr == 'ts':
            surrogate = np.zeros((NN, Nsurr, T))
            for k in range(NN):
                for n in range(Nsurr):
                    surrogate[k, n] = timeshifted(z[k], -(n + dimEmb + 1))
        elif surr == 'iaaft':
            surrogate = np.zeros((NN, Nsurr, T))
            for k in range(NN):
                for n in range(Nsurr):
                    surrogate[k, n], _, _ = iaaft(z[k])

        for i in channels:
            EmbdDumm = embed(z[i], dimEmb + 1, tau)
            Xtau     = EmbdDumm[:, :-1]
            for j in channels:
                if i == j:
                    continue
                ptedumold = -np.inf
                for n in range(Nsurr):
                    Yembd = embed(surrogate[j, n], dimEmb + 1, tau)
                    Y     = Yembd[:, -1]
                    Ytau  = Yembd[:, :-1]

                    XtYt  = np.concatenate((Xtau,   Ytau),  axis=1)
                    YYt   = np.concatenate((Y[:, np.newaxis], Ytau), axis=1)
                    YYtXt = np.concatenate((YYt, Xtau), axis=1)

                    if dimEmb > 1:
                        ptedum = (
                            np.linalg.det(np.cov(XtYt.T))
                            * np.linalg.det(np.cov(YYt.T))
                        ) / (
                            np.linalg.det(np.cov(YYtXt.T))
                            * np.linalg.det(np.cov(Ytau.T))
                        )
                    else:
                        ptedum = (
                            _det_safe(np.cov(XtYt.T))
                            * _det_safe(np.cov(YYt.T))
                        ) / (
                            _det_safe(np.cov(YYtXt.T))
                            * _det_safe(np.cov(Ytau.T))
                        )
                    if ptedum > ptedumold:
                        ptedumold = ptedum
                ptesurr[i, j] = 0.5 * np.log(max(ptedumold, 1e-12))

    return pte, ptesurr
