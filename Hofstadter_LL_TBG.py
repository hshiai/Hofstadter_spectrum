"""
Calculate the Hofstadter spectrum for TBG in the continuum LL basis.

This version is written directly from the new q*a1 convention note:
    B = chi * B_abs * ez, chi = +/- 1
    A = chi * B_abs * x * ey
    Phi/Phi0 = chi * p/q, |Phi|/Phi0 = p/q
    m1 = (q/p) a1,     m2 = a2
    g1 = (p/q) b1,     g2 = b2
    P1 = b1/q = g1/p,  P2 = b2 = g2
    |eta,l,alpha,n,r,k> = exp(i eta Gamma.r) chi_{l alpha}
        |psi^chi_{n, ((k1+r)/p) g1 + k2 g2}>, r=0,...,p-1.

The implementation keeps the old code style but updates every formula to the
new convention, including chi=+/-1.  The Hamiltonian is dense; only band-window
routines use subset diagonalization.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from scipy.special import gammaln

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False


class structure:
    # Basic TBG parameters.
    theta_d = 1.5
    pi = np.pi
    a0 = 2.46       # [Angstrom]
    theta_r = theta_d / 180.0 * pi
    Ltheta = a0 / (2.0 * np.sin(theta_r / 2.0))

    # Primitive vectors in real and reciprocal space.
    # a2 is chosen along y, as in the continuum Hofstadter note.
    a1 = Ltheta * np.array([0.5 * np.sqrt(3.0), 0.5])
    a2 = Ltheta * np.array([0.0, 1.0])
    b1 = 4.0 * pi / (np.sqrt(3.0) * Ltheta) * np.array([1.0, 0.0])
    b2 = 4.0 * pi / (np.sqrt(3.0) * Ltheta) * np.array([-0.5, 0.5 * np.sqrt(3.0)])

    # Dirac points after gauging out Gamma: bar{K}_1, bar{K}_2.
    K1 = 4.0 * pi / (3.0 * Ltheta) * np.array([0.5 * np.sqrt(3.0), -0.5])
    K2 = 4.0 * pi / (3.0 * Ltheta) * np.array([0.5 * np.sqrt(3.0), +0.5])

    hbarvF = 5.944  # [eV Angstrom]

    # Moire unit-cell area.
    S0 = 0.5 * np.sqrt(3.0) * Ltheta**2

    # Pauli matrices.
    Pauli = np.zeros((4, 2, 2), complex)
    Pauli[0] = np.eye(2)
    Pauli[1] = np.array([[0.0, 1.0], [1.0, 0.0]])
    Pauli[2] = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    Pauli[3] = np.array([[1.0, 0.0], [0.0, -1.0]])

    # Tunneling matrices for eta=+.
    u1 = 0.11    # [eV]
    u0 = 0.8 * u1
    omega = np.exp(1j * 2.0 * pi / 3.0)
    T1 = np.array([[u0, u1], [u1, u0]], complex)
    T2 = np.array([[u0, u1 * omega], [u1 * np.conj(omega), u0]], complex)
    T3 = np.array([[u0, u1 * np.conj(omega)], [u1 * omega, u0]], complex)
    Pot_coef_plus = np.array([T1, T2, T3], complex)

    # U_+(r) = T1 + T2 exp[-i b2.r] + T3 exp[-i(b1+b2).r].
    # U_eta is obtained by multiplying the reciprocal vectors by eta and using T_eta.
    Pot_list_plus = np.array([[0, 0], [0, -1], [-1, -1]], int)

    # Landau-level oscillator cutoff.  The physical basis keeps one extra oscillator
    # on the zero-mode sublattice and deletes one top oscillator on the other sublattice.
    NLL = 51


def _check_pm(name, value):
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(f'{name} must be +1 or -1')
    return value


def Generate_pq_list(qmax):
    pq_list = []
    for q in range(1, qmax + 1):
        for p in range(1, q + 1):
            if np.gcd(p, q) == 1:
                pq_list.append(np.array([p, q], int))
    return np.array(pq_list, int)


def Cal_lB(p, q):
    return np.sqrt(structure.S0 / (2.0 * structure.pi) * q / p)


def Cal_magnetic_vectors(p, q):
    # q*a1 convention.
    m1 = (q / p) * structure.a1
    m2 = structure.a2.copy()
    g1 = (p / q) * structure.b1
    g2 = structure.b2.copy()
    P1 = structure.b1 / q
    P2 = structure.b2.copy()
    return m1, m2, g1, g2, P1, P2


def Cal_delete_indices(valley, chi):
    """Base indices to delete before tensoring with r.

    Base order is (layer, sublattice, oscillator n), dimension 4*NLL:
        layer 1 A: [0, NLL)
        layer 1 B: [NLL, 2*NLL)
        layer 2 A: [2*NLL, 3*NLL)
        layer 2 B: [3*NLL, 4*NLL)

    From the note, the zero mode is on A if eta=-chi and on B if eta=chi.
    We keep the zero-mode sublattice larger by one, i.e. delete the top oscillator
    of the opposite sublattice in both layers.
    """
    valley = _check_pm('valley', valley)
    chi = _check_pm('chi', chi)
    N = structure.NLL
    if valley == -chi:
        # zero mode on A; delete top B oscillator in both layers
        return np.array([2 * N - 1, 4 * N - 1], int)
    # zero mode on B; delete top A oscillator in both layers
    return np.array([1 * N - 1, 3 * N - 1], int)


def Cal_keep_indices_base(valley, chi):
    N = structure.NLL
    delete = set(Cal_delete_indices(valley, chi).tolist())
    return np.array([i for i in range(4 * N) if i not in delete], int)


def Cal_keep_indices_full(p, valley, chi):
    keep_base = Cal_keep_indices_base(valley, chi)
    keep = []
    for ib in keep_base:
        keep.extend(range(ib * p, (ib + 1) * p))
    return np.array(keep, int)


def Cal_T_matrices(valley):
    valley = _check_pm('valley', valley)
    if valley == +1:
        return structure.Pot_coef_plus.copy()
    return np.conj(structure.Pot_coef_plus.copy())


def Cal_Pot_list(valley):
    valley = _check_pm('valley', valley)
    return valley * structure.Pot_list_plus.copy()


def Cal_Tmat(p, q, valley=+1, chi=+1):
    """Full kinetic matrix before r tensor and LL-cut deletion.

    Implements the note formula
        h^eta(pi) = i hbar vF/(sqrt(2) lB)
            [[0, (eta-chi) a^dag - (eta+chi) a],
             [(eta+chi) a^dag + (chi-eta) a, 0]].
    The common Gamma shift has been gauged out, leaving -hbar vF eta Kbar_l.sigma_eta.
    """
    valley = _check_pm('valley', valley)
    chi = _check_pm('chi', chi)

    lB = Cal_lB(p, q)
    hbarvF = structure.hbarvF
    Pauli = structure.Pauli
    NLL = structure.NLL

    sig_p = 0.5 * (Pauli[1] + 1j * Pauli[2])  # upper-right block
    sig_n = 0.5 * (Pauli[1] - 1j * Pauli[2])  # lower-left block
    layer_1 = 0.5 * (Pauli[0] + Pauli[3])
    layer_2 = 0.5 * (Pauli[0] - Pauli[3])

    n_list = np.arange(NLL, dtype=float)
    sqrtn_list = np.sqrt(n_list)
    a = np.diag(sqrtn_list[1:], +1).astype(complex)
    adag = np.diag(sqrtn_list[1:], -1).astype(complex)

    upper_right = (valley - chi) * adag - (valley + chi) * a
    lower_left = (valley + chi) * adag + (chi - valley) * a
    Dmat = 1j * hbarvF / (np.sqrt(2.0) * lB) * (
        np.kron(sig_p, upper_right) + np.kron(sig_n, lower_left)
    )

    K1 = structure.K1
    K2 = structure.K2
    # bar{K}.sigma_eta = eta K_x sigma_x + K_y sigma_y.
    K1_dot_sigma = np.kron(hbarvF * (valley * K1[0] * Pauli[1] + K1[1] * Pauli[2]), np.eye(NLL))
    K2_dot_sigma = np.kron(hbarvF * (valley * K2[0] * Pauli[1] + K2[1] * Pauli[2]), np.eye(NLL))

    Tmat = (
        np.kron(Pauli[0], Dmat)
        - valley * np.kron(layer_1, K1_dot_sigma)
        - valley * np.kron(layer_2, K2_dot_sigma)
    )
    return Tmat


def Cal_laguerre(x):
    # Return laguerre_mat[n, alpha] = L_n^alpha(x), for n, alpha < NLL.
    NLL = structure.NLL
    alpha = np.arange(NLL, dtype=float)
    laguerre_mat = np.zeros((NLL, NLL), float)
    laguerre_mat[0, :] = 1.0
    if NLL > 1:
        laguerre_mat[1, :] = 1.0 + alpha - x
    for n in range(1, NLL - 1):
        Ln = laguerre_mat[n, :]
        Ln1 = laguerre_mat[n - 1, :]
        laguerre_mat[n + 1, :] = ((2 * n + 1 + alpha - x) * Ln - (n + alpha) * Ln1) / (n + 1)
    return laguerre_mat


def Cal_F_from_qvec(qvec, p, q, chi):
    """Return F_{n' n}(Q_chi(qvec)), with rows n', columns n.

    Q_chi(q) = (q_x + i chi q_y) lB / sqrt(2).
    """
    chi = _check_pm('chi', chi)
    NLL = structure.NLL
    lB = Cal_lB(p, q)
    Q = (qvec[0] + 1j * chi * qvec[1]) * lB / np.sqrt(2.0)
    Qnorm = np.abs(Q)
    if Qnorm < 1.0e-14:
        return np.eye(NLL, dtype=complex)

    Qang = np.angle(Q)
    lag = Cal_laguerre(Qnorm**2)
    Fmat = np.zeros((NLL, NLL), complex)
    logQ = np.log(Qnorm)
    for np1 in range(NLL):       # row n'
        for n in range(NLL):     # column n
            if np1 <= n:
                diff = n - np1
                Fmat[np1, n] = (
                    np.exp(1j * (0.5 * np.pi + Qang) * diff)
                    * np.exp(0.5 * (gammaln(np1 + 1) - gammaln(n + 1) - Qnorm**2) + diff * logQ)
                    * lag[np1, diff]
                )
            else:
                diff = np1 - n
                Fmat[np1, n] = (
                    np.exp(1j * (0.5 * np.pi - Qang) * diff)
                    * np.exp(0.5 * (gammaln(n + 1) - gammaln(np1 + 1) - Qnorm**2) + diff * logQ)
                    * lag[n, diff]
                )
    return Fmat


def Cal_Fmat(p, q, valley=+1, chi=+1):
    Pot_list = Cal_Pot_list(valley)
    Fmat = np.zeros((len(Pot_list), structure.NLL, structure.NLL), complex)
    for ihop, (G1, G2) in enumerate(Pot_list):
        Gvec = G1 * structure.b1 + G2 * structure.b2
        Fmat[ihop] = Cal_F_from_qvec(Gvec, p, q, chi)
    return Fmat


def Cal_Rmat(k1, k2, p, q, G1, G2, chi):
    """r-space matrix for exp[i(G1*b1+G2*b2).r].

    Note formula:
        R_{r' r} = delta_{r', r + G1*q mod p}
            exp[i chi 2pi (k1+r) G2/p - i chi 2pi (q/p) G1 (k2-G2/2)].
    """
    chi = _check_pm('chi', chi)
    G1 = int(G1)
    G2 = int(G2)
    r_list = np.arange(p, dtype=int)
    rp_list = (r_list + G1 * q) % p
    phase = np.exp(
        1j * chi * 2.0 * np.pi * (k1 + r_list) * G2 / p
        - 1j * chi * 2.0 * np.pi * (q / p) * G1 * (k2 - 0.5 * G2)
    )
    Rmat = np.zeros((p, p), complex)
    Rmat[rp_list, r_list] = phase
    return Rmat


def Cal_Hamk(k1, k2, p, q, Tmat=None, Fmat=None, valley=+1, chi=+1):
    valley = _check_pm('valley', valley)
    chi = _check_pm('chi', chi)
    NLL = structure.NLL
    Pauli = structure.Pauli

    if Tmat is None:
        Tmat = Cal_Tmat(p, q, valley, chi)
    if Fmat is None:
        Fmat = Cal_Fmat(p, q, valley, chi)

    Pot_list = Cal_Pot_list(valley)
    Pot_coef = Cal_T_matrices(valley)

    # Kinetic term: full order is (layer, sublattice, n, r).
    H_full = np.kron(Tmat, np.eye(p, dtype=complex))

    # Interlayer block U: layer 2 -> layer 1.
    U_internal = np.zeros((2 * NLL * p, 2 * NLL * p), complex)
    for ihop, (G1, G2) in enumerate(Pot_list):
        R = Cal_Rmat(k1, k2, p, q, G1, G2, chi)
        U_internal += np.kron(Pot_coef[ihop], np.kron(Fmat[ihop], R))

    layer_12 = 0.5 * (Pauli[1] + 1j * Pauli[2])
    U_full = np.kron(layer_12, U_internal)
    H_full = H_full + U_full + U_full.conj().T

    # Delete the extra top oscillator on the non-zero-mode sublattice in both layers.
    keep = Cal_keep_indices_full(p, valley, chi)
    H = H_full[np.ix_(keep, keep)]
    return 0.5 * (H + H.conj().T)


def _eigh_subset(H, nb_start=None, nb_end=None, eigvals_only=False):
    if nb_start is None or nb_end is None:
        if eigvals_only:
            return eigvalsh(H, check_finite=False, overwrite_a=True)
        return eigh(H, check_finite=False, overwrite_a=True, driver='evr')
    subset = [int(nb_start), int(nb_end) - 1]
    return eigh(H, eigvals_only=eigvals_only, subset_by_index=subset,
                check_finite=False, overwrite_a=True, driver='evr')


def _parallel_map(func, tasks, n_jobs):
    if n_jobs == 1 or not HAS_JOBLIB:
        return [func(*task) for task in tasks]
    return Parallel(n_jobs=n_jobs, prefer='processes')(delayed(func)(*task) for task in tasks)


def Plot_band(p, q, nb_start, nb_end, valley=+1, chi=+1, num_k1=30, num_k2=30, n_jobs=1):
    Tmat = Cal_Tmat(p, q, valley, chi)
    Fmat = Cal_Fmat(p, q, valley, chi)

    k1_list = np.linspace(0.0, 1.0, num_k1, endpoint=False)
    k2_list = np.linspace(0.0, 1.0, num_k2, endpoint=False)
    tasks = [(float(k1), float(k2)) for k1 in k1_list for k2 in k2_list]

    def solve(k1, k2):
        H = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, valley, chi)
        return _eigh_subset(H, nb_start, nb_end, eigvals_only=True)

    vals = _parallel_map(solve, tasks, n_jobs)
    Eband = np.array(vals, float).reshape(num_k1, num_k2, nb_end - nb_start)

    for ik2 in range(num_k2):
        plt.plot(k1_list, Eband[:, ik2, :])
    plt.xlabel('k1')
    plt.title(f'TBG valley={valley}, chi={chi}, flux=chi*p/q={chi}*{p}/{q} along P1')
    plt.show()

    for ik1 in range(num_k1):
        plt.plot(k2_list, Eband[ik1, :, :])
    plt.xlabel('k2')
    plt.title(f'TBG valley={valley}, chi={chi}, flux=chi*p/q={chi}*{p}/{q} along P2')
    plt.show()

    return Eband


def Collect_spectrum(qmax, numk, valley=+1, chi=+1, n_jobs=1, reduced_kmesh=True):
    """Collect full spectra with the old public interface plus optional reduced mesh.

    No energy-window/band-window input is used here. Energy cuts are applied only
    in Plot_butterfly.

    In the q*a1 convention, t(a1) maps k2 -> k2 - chi*p/q while leaving k1
    unchanged. Therefore the spectrum is q-fold repeated along k2, and the
    reduced mesh may sample k2 in [0, 1/q). This keeps the same spectrum while
    avoiding redundant k points. Set reduced_kmesh=False to sample the full SBZ.
    """
    pq_list = Generate_pq_list(qmax)
    print('totally', len(pq_list), '(p, q) pairs')
    phi_list = []
    E_list = []

    for ipq, (p, q) in enumerate(pq_list):
        t1 = time.time()
        p = int(p)
        q = int(q)
        Tmat = Cal_Tmat(p, q, valley, chi)
        Fmat = Cal_Fmat(p, q, valley, chi)

        k1_list = np.linspace(0.0, 1.0, numk, endpoint=False)
        if reduced_kmesh:
            k2_list = np.linspace(0.0, 1.0 / q, numk, endpoint=False)
        else:
            k2_list = np.linspace(0.0, 1.0, numk, endpoint=False)
        tasks = [(float(k1), float(k2)) for k1 in k1_list for k2 in k2_list]

        def solve(k1, k2):
            H = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, valley, chi)
            return _eigh_subset(H, eigvals_only=True)

        vals = _parallel_map(solve, tasks, n_jobs)
        E_ipq = np.array(vals, float).reshape(-1)

        phi_list.append(chi * p / q)
        E_list.append(E_ipq)
        mesh_note = 'reduced k2 in [0,1/q)' if reduced_kmesh else 'full SBZ'
        print(ipq, '-th pair: (p, q, valley, chi) =', p, q, valley, chi,
              mesh_note, 'finished, used', time.time() - t1, 'seconds')

    return phi_list, E_list


def Plot_butterfly(phi_list, E_list, Ecut_lower, Ecut_upper):
    for phi, E in zip(phi_list, E_list):
        E = np.asarray(E)
        E = E[(E > Ecut_lower) & (E < Ecut_upper)]
        plt.plot(phi * np.ones(E.size), E, 'k.', markersize=0.8)
    plt.xlabel(r'$\Phi/\Phi_0$')
    plt.ylim([Ecut_lower, Ecut_upper])
    plt.show()


def Cal_X_overlap(dk1, dk2, k1, k2, p, q, valley=+1, chi=+1):
    """Return X^chi(k+dk,k) in the reduced basis.

    X_{l' alpha' n' r', l alpha n r} = delta_ll' delta_aa' delta_rr'
      exp[-i chi pi/p dk1 (k2'+k2)] F_{n'n}(Q_chi(dk1 P1+dk2 P2)).
    """
    _, _, _, _, P1, P2 = Cal_magnetic_vectors(p, q)
    qvec = dk1 * P1 + dk2 * P2
    Fdelta = Cal_F_from_qvec(qvec, p, q, chi)
    phase = np.exp(-1j * chi * np.pi * dk1 * (2.0 * k2 + dk2) / p)

    X_base = np.kron(np.eye(4, dtype=complex), Fdelta)
    keep_base = Cal_keep_indices_base(valley, chi)
    X_base = X_base[np.ix_(keep_base, keep_base)]
    X = phase * np.kron(X_base, np.eye(p, dtype=complex))
    return X


def Cal_Chern_number(p, q, nb_start, nb_end, valley=+1, chi=+1, numk=10, n_jobs=1):
    Tmat = Cal_Tmat(p, q, valley, chi)
    Fmat = Cal_Fmat(p, q, valley, chi)
    dim = (4 * structure.NLL - 2) * p
    Nb = nb_end - nb_start

    dk1 = 1.0 / numk
    dk2 = 1.0 / numk
    k1_base = np.linspace(0.0, 1.0, numk, endpoint=False)
    k2_base = np.linspace(0.0, 1.0, numk, endpoint=False)
    k1_list = np.concatenate([k1_base, [1.0, 1.0 + dk1]])
    k2_list = np.concatenate([k2_base, [1.0, 1.0 + dk2]])

    tasks = [(ik1, ik2, float(k1), float(k2))
             for ik1, k1 in enumerate(k1_list)
             for ik2, k2 in enumerate(k2_list)]

    def solve(ik1, ik2, k1, k2):
        H = Cal_Hamk(k1, k2, p, q, Tmat, Fmat, valley, chi)
        E, P = _eigh_subset(H, nb_start, nb_end, eigvals_only=False)
        return ik1, ik2, P

    results = _parallel_map(solve, tasks, n_jobs)
    Psi = np.zeros((numk + 2, numk + 2, dim, Nb), complex)
    for ik1, ik2, P in results:
        Psi[ik1, ik2] = P
    print('finished wavefunctions')

    Umat = np.zeros((numk + 1, numk + 1, 2), complex)
    X2 = Cal_X_overlap(0.0, dk2, 0.0, 0.0, p, q, valley, chi)
    X1_cache = {}

    for ik1 in range(numk + 1):
        for ik2 in range(numk + 1):
            k2 = k2_list[ik2]
            if k2 not in X1_cache:
                X1_cache[k2] = Cal_X_overlap(dk1, 0.0, k1_list[ik1], k2, p, q, valley, chi)
            X1 = X1_cache[k2]
            P0 = Psi[ik1, ik2]
            P1 = Psi[ik1 + 1, ik2]
            P2 = Psi[ik1, ik2 + 1]

            M1 = P0.conj().T @ X1 @ P1
            M2 = P0.conj().T @ X2 @ P2
            d1 = np.linalg.det(M1)
            d2 = np.linalg.det(M2)
            if abs(d1) < 1e-14 or abs(d2) < 1e-14:
                raise RuntimeError('Near-zero overlap determinant; refine k mesh or avoid a band crossing.')
            Umat[ik1, ik2, 0] = d1 / abs(d1)
            Umat[ik1, ik2, 1] = d2 / abs(d2)

    Fsum = 0.0j
    for ik1 in range(numk):
        for ik2 in range(numk):
            loop = np.log(Umat[ik1, ik2, 0] * Umat[ik1 + 1, ik2, 1]
                          / (Umat[ik1, ik2 + 1, 0] * Umat[ik1, ik2, 1]))
            imag = np.imag(loop)
            if imag >= np.pi:
                imag -= 2.0 * np.pi
            elif imag < -np.pi:
                imag += 2.0 * np.pi
            Fsum += 1j * imag

    Chern = Fsum / (2.0j * np.pi)
    print('Chern number =', Chern)
    return Chern


if __name__ == '__main__':
    valley = +1
    chi = +1

    # Butterfly example. This computes the full spectrum on the reduced k mesh; plot_butterfly applies the energy cut.
    phi_list, E_list = Collect_spectrum(12, 3, valley=valley, chi=chi, n_jobs=1)
    Plot_butterfly(phi_list, E_list, -0.2, 0.2)

    # p = 1
    # q = 1
    # nb_start = (2 * structure.NLL - 1) * p - 1
    # nb_end = (2 * structure.NLL - 1) * p + 1
    # Plot_band(p, q, nb_start, nb_end, valley=valley, chi=chi, num_k1=3, num_k2=3, n_jobs=1)
    # Chern = Cal_Chern_number(p, q, nb_start, nb_end, valley=valley, chi=chi, numk=6, n_jobs=1)
