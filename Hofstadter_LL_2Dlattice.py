"""
calculate Hofstadter spectrum for a generic 2D lattice in continuum model
(no sublattice), using the q*a1 convention and signed field chi = +/- 1.

Convention:
    B = chi * B_abs * ez,  B_abs > 0
    A = chi * B_abs * x * ey
    Phi/Phi0 = chi * p/q
    m1 = (q/p) a1, m2 = a2
    g1 = (p/q) b1, g2 = b2
    L1 = q a1, L2 = a2
    P1 = b1/q = g1/p, P2 = b2 = g2
    |n,r,k> = |psi^chi_{n, ((k1+r)/p) g1 + k2 g2}>, r=0,...,p-1

Plane-wave matrix element for G = G1*b1 + G2*b2:
    <n'r'k| exp(i G.r) | n r k>
      = F_{n'n}(Q_chi(G))
        exp[i chi 2pi (k1+r)G2/p - i chi 2pi (q/p)G1 (k2-G2/2)]
        delta^{mod p}_{r', r+G1 q}

Created on Jun 19 2023
Rewritten on Jun 15 2026
@author: shihao, revised by ChatGPT
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.special import gammaln


class structure:
    # Unit length is |a2|.  We fix a2 parallel to y, i.e. a2=(0,1).
    # Unit energy is hbar^2/(m |a2|^2).
    pi = np.pi

    # primitive lattice vectors in real space: a2=(0,1), a1x>0
    a1 = np.array([0.5 * np.sqrt(3.0), 0.5], dtype=float)
    a2 = np.array([0.0, 1.0], dtype=float)

    # reciprocal lattice vectors, ai.bj=2pi delta_ij
    b1 = 2.0 * pi * np.array([1.0 / a1[0], 0.0], dtype=float)
    b2 = 2.0 * pi * np.array([-a1[1] / a1[0], 1.0], dtype=float)

    # unit-cell area S0=(a1 x a2)_z = a1x
    S0 = a1[0]

    # Potential term: V(r)=sum_G Pot_coef[G] exp(i G.r)+h.c.
    # G = G1*b1 + G2*b2, with integer pair (G1,G2)
    Pot_list = np.array([[1, 0], [0, 1], [1, 1]], dtype=int)
    Pot_coef = np.array([-3.0, -3.0, -3.0], dtype=complex)

    # Landau-level cutoff
    NLL = 51


def validate_chi(chi):
    chi = int(chi)
    if chi not in (-1, 1):
        raise ValueError("chi must be +1 or -1")
    return chi


def Generate_pq_list(qmax):
    pq_list = []
    for q in range(1, qmax + 1):
        for p in range(1, q + 1):
            if np.gcd(p, q) == 1:
                pq_list.append(np.array([p, q], dtype=int))
    return np.array(pq_list, dtype=int)


def magnetic_length(p, q):
    # |Phi|/Phi0 = p/q, lB^2 = S0/(2pi) * q/p
    return np.sqrt(structure.S0 / (2.0 * structure.pi) * q / p)


def Cal_Tmat(p, q):
    lB = magnetic_length(p, q)
    n = np.arange(structure.NLL, dtype=float)
    return np.diag((n + 0.5) / lB**2)


def Cal_laguerre(x):
    """Return matrix L[n, alpha] = L_n^alpha(x), n,alpha < NLL."""
    NLL = structure.NLL
    alpha = np.arange(NLL, dtype=float)
    laguerre_mat = np.zeros((NLL, NLL), dtype=float)
    laguerre_mat[0, :] = 1.0
    if NLL > 1:
        laguerre_mat[1, :] = 1.0 + alpha - x
    for n in range(1, NLL - 1):
        Ln = laguerre_mat[n, :]
        Lnm1 = laguerre_mat[n - 1, :]
        laguerre_mat[n + 1, :] = ((2 * n + 1 + alpha - x) * Ln - (n + alpha) * Lnm1) / (n + 1)
    return laguerre_mat


def FormFactor_from_qvec(qvec, p, q, chi):
    """Matrix F_{n' n}(Q_chi(qvec)), rows n', columns n."""
    chi = validate_chi(chi)
    lB = magnetic_length(p, q)
    NLL = structure.NLL
    Qcomp = (qvec[0] + 1j * chi * qvec[1]) * lB / np.sqrt(2.0)
    Qnorm = np.abs(Qcomp)

    # Avoid log(0).  For Q=0, exp(i q.r)=1 in LL space.
    if Qnorm < 1.0e-14:
        return np.eye(NLL, dtype=complex)

    Qangl = np.angle(Qcomp)
    laguerre = Cal_laguerre(Qnorm**2)
    Fmat = np.zeros((NLL, NLL), dtype=complex)
    for np_ in range(NLL):
        for n in range(NLL):
            if np_ <= n:
                diff = n - np_
                Fmn = np.exp(1j * (0.5 * np.pi + Qangl) * diff) * \
                      np.exp(0.5 * (gammaln(np_ + 1) - gammaln(n + 1) - Qnorm**2)
                             + diff * np.log(Qnorm)) * laguerre[np_, diff]
            else:
                diff = np_ - n
                Fmn = np.exp(1j * (0.5 * np.pi - Qangl) * diff) * \
                      np.exp(0.5 * (gammaln(n + 1) - gammaln(np_ + 1) - Qnorm**2)
                             + diff * np.log(Qnorm)) * laguerre[n, diff]
            Fmat[np_, n] = Fmn
    return Fmat


def Cal_Fmat(p, q, chi=1):
    chi = validate_chi(chi)
    Pot_list = structure.Pot_list
    Nhop = Pot_list.shape[0]
    Fmat = np.zeros((Nhop, structure.NLL, structure.NLL), dtype=complex)
    for ihop, (G1, G2) in enumerate(Pot_list):
        Gvec = G1 * structure.b1 + G2 * structure.b2
        Fmat[ihop] = FormFactor_from_qvec(Gvec, p, q, chi)
    return Fmat


def Cal_Rmat_G(k1, k2, p, q, chi, G1, G2):
    """r-space matrix for exp[i(G1*b1+G2*b2).r] in qa1 convention."""
    chi = validate_chi(chi)
    r = np.arange(p, dtype=int)
    rc, rl = np.meshgrid(r, r)  # columns: ket r; rows: bra r'
    mask = ((rl - rc - G1 * q) % p == 0)
    phase = np.exp(
        1j * chi * 2.0 * np.pi * (k1 + rc) * G2 / p
        - 1j * chi * 2.0 * np.pi * (q / p) * G1 * (k2 - 0.5 * G2)
    )
    return mask.astype(complex) * phase


def Cal_Hamk(k1, k2, p, q, Tmat, Fmat, chi=1):
    chi = validate_chi(chi)
    NLL = structure.NLL
    T = np.kron(Tmat, np.eye(p, dtype=complex))
    V = np.zeros((p * NLL, p * NLL), dtype=complex)

    for ihop, (G1, G2) in enumerate(structure.Pot_list):
        Rmat = Cal_Rmat_G(k1, k2, p, q, chi, int(G1), int(G2))
        V_G = structure.Pot_coef[ihop] * np.kron(Fmat[ihop], Rmat)
        V += V_G + V_G.conj().T

    return T + V


def Plot_band(p, q, nb_start, nb_end, chi=1):
    chi = validate_chi(chi)
    num_b = nb_end - nb_start
    Tmat = Cal_Tmat(p, q)
    Fmat = Cal_Fmat(p, q, chi)

    num_k1 = 80
    num_k2 = 30
    k1_list = np.linspace(0.0, 1.0, num_k1, endpoint=False)
    k2_list = np.linspace(0.0, 1.0, num_k2, endpoint=False)

    Eband = np.zeros((num_k1, num_k2, num_b), dtype=float)
    for ik1, k1 in enumerate(k1_list):
        for ik2, k2 in enumerate(k2_list):
            Hamk = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, chi)
            Ek, _ = eigh(Hamk)
            Eband[ik1, ik2, :] = Ek[nb_start:nb_end]

    for ik2 in range(num_k2):
        plt.plot(k1_list, Eband[:, ik2, :])
    plt.xlabel(r'$k_1$')
    plt.title(f'band at signed flux chi*p/q = {chi}*{p}/{q}, along P1')
    plt.show()

    for ik1 in range(num_k1):
        plt.plot(k2_list, Eband[ik1, :, :])
    plt.xlabel(r'$k_2$')
    plt.title(f'band at signed flux chi*p/q = {chi}*{p}/{q}, along P2')
    plt.show()


def Collect_spectrum(qmax, numk, chi=1, use_qfold_reduction=True):
    chi = validate_chi(chi)
    pq_list = Generate_pq_list(qmax)
    print('totally', len(pq_list), '(p, q) pairs')
    phi_list = []
    E_list = []

    for ipq, (p, q) in enumerate(pq_list):
        t1 = time.time()
        if use_qfold_reduction:
            # q-fold degeneracy is along k2 in the q*a1 convention.
            k1_list = np.linspace(0.0, 1.0, numk, endpoint=False)
            k2_list = np.linspace(0.0, 1.0 / q, numk, endpoint=False)
        else:
            k1_list = np.linspace(0.0, 1.0, numk, endpoint=False)
            k2_list = np.linspace(0.0, 1.0, numk, endpoint=False)

        Tmat = Cal_Tmat(p, q)
        Fmat = Cal_Fmat(p, q, chi)
        E_ipq = np.zeros((numk, numk, p * structure.NLL), dtype=float)

        for ik1, k1 in enumerate(k1_list):
            for ik2, k2 in enumerate(k2_list):
                Hamk = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, chi)
                E, _ = eigh(Hamk)
                E_ipq[ik1, ik2, :] = E

        phi_list.append(chi * p / q)
        E_list.append(E_ipq.reshape(-1))
        print(ipq, '-th pair: (p, q, chi) =', p, q, chi, 'finished, used', time.time() - t1, 'seconds')

    return phi_list, E_list


def Plot_butterfly(phi_list, E_list, Ecut_lower, Ecut_upper):
    phi_arr = np.asarray(phi_list, dtype=float)
    for phi, E in zip(phi_arr, E_list):
        E = np.asarray(E)
        E = E[(E > Ecut_lower) & (E < Ecut_upper)]
        plt.plot(phi * np.ones(E.size), E, 'k.', markersize=0.8)
    plt.ylim([Ecut_lower, Ecut_upper])
    if phi_arr.size > 0:
        pad = 0.03 * max(1.0, np.max(np.abs(phi_arr)))
        plt.xlim([np.min(phi_arr) - pad, np.max(phi_arr) + pad])
    plt.xlabel(r'$\Phi/\Phi_0$')
    plt.show()


def Cal_Chern_number(p, q, nb_start, nb_end, chi=1, numk=20):
    chi = validate_chi(chi)
    Nb = nb_end - nb_start
    NLL = structure.NLL
    dim = p * NLL

    k1_base = np.linspace(0.0, 1.0, numk, endpoint=False)
    k2_base = np.linspace(0.0, 1.0, numk, endpoint=False)
    dk1 = 1.0 / numk
    dk2 = 1.0 / numk
    k1_list = np.concatenate([k1_base, [1.0, 1.0 + dk1]])
    k2_list = np.concatenate([k2_base, [1.0, 1.0 + dk2]])

    Tmat = Cal_Tmat(p, q)
    Fmat = Cal_Fmat(p, q, chi)

    Psi_list = np.zeros((numk + 2, numk + 2, dim, Nb), dtype=complex)
    for ik1, k1 in enumerate(k1_list):
        for ik2, k2 in enumerate(k2_list):
            Hamk = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, chi)
            _, Pk = eigh(Hamk)
            Psi_list[ik1, ik2, :, :] = Pk[:, nb_start:nb_end]
    print('finished wavefunctions')

    # Link overlap matrices X(k+dk_i,k).  The lifted k' is used.
    F_delta1 = FormFactor_from_qvec(dk1 * (structure.b1 / q), p, q, chi)
    F_delta2 = FormFactor_from_qvec(dk2 * structure.b2, p, q, chi)
    Inner2 = np.kron(F_delta2, np.eye(p, dtype=complex))

    Umat = np.zeros((numk + 1, numk + 1, 2), dtype=complex)
    for ik1 in range(numk + 1):
        for ik2 in range(numk + 1):
            k2 = k2_list[ik2]
            Psi_k = Psi_list[ik1, ik2]
            Psi_k1 = Psi_list[ik1 + 1, ik2]
            Psi_k2 = Psi_list[ik1, ik2 + 1]

            phase1 = np.exp(-1j * chi * np.pi * dk1 * (2.0 * k2) / p)
            Inner1 = np.kron(F_delta1, phase1 * np.eye(p, dtype=complex))

            Det_k1 = np.linalg.det(Psi_k.conj().T @ Inner1 @ Psi_k1)
            Det_k2 = np.linalg.det(Psi_k.conj().T @ Inner2 @ Psi_k2)
            Umat[ik1, ik2, 0] = Det_k1 / np.abs(Det_k1)
            Umat[ik1, ik2, 1] = Det_k2 / np.abs(Det_k2)
    
    FFmat = np.zeros((numk, numk), dtype=complex)
    for ik1 in range(numk):
        for ik2 in range(numk):
            loop = np.log(Umat[ik1, ik2, 0] * Umat[ik1 + 1, ik2, 1]
                          / (Umat[ik1, ik2 + 1, 0] * Umat[ik1, ik2, 1]))
            imag = np.imag(loop)
            if imag >= np.pi:
                imag -= 2.0 * np.pi
            elif imag < -np.pi:
                imag += 2.0 * np.pi
            FFmat[ik1, ik2] = 1j * imag

    Chern = np.sum(FFmat) / (2.0j * np.pi)
    print('Chern number =', Chern)
    return Chern


if __name__ == '__main__':
    p = 1
    q = 5
    chi = -1
    nb_start = 0
    nb_end = 2
    Plot_band(p, q, nb_start, nb_end, chi)
    Chern = Cal_Chern_number(p, q, nb_start, nb_end, chi, numk=20)
    
    phi_list, E_list = Collect_spectrum(qmax=13, numk=3, chi=chi, use_qfold_reduction=True)
    Plot_butterfly(phi_list, E_list, Ecut_lower=-3, Ecut_upper=3.0)
