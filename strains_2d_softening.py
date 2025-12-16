
# strains_2d_softening.py
# 2D small-strain FE (Q4) + local J2 plasticity with linear softening (H < 0) in plane strain.
# Purpose: demonstrate pathological mesh sensitivity (local softening -> localization width ~ 1 element).
#
# Dependencies: numpy, scipy, matplotlib
#
# Notes:
# - Displacement-controlled loading (prescribed uy on top boundary) to follow post-peak.
# - Adds a small "weak band" (slightly reduced yield stress) to trigger localization reproducibly.
# - Tangent at Gauss points is computed by finite differences (robust, easier to verify).

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dataclasses import dataclass

# -------------------------
# Mesh utilities (Q4 quad)
# -------------------------
def structured_quad_mesh(nx: int, ny: int, L: float, H: float):
    """
    Structured mesh of Q4 elements on [0,L]x[0,H].

    Returns
    -------
    X : (nnode, 2) float
        Node coordinates.
    conn : (nelem, 4) int
        Element connectivity (counter-clockwise).
    """
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    X = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def nid(i, j):  # i in [0,nx], j in [0,ny]
        return j * (nx + 1) + i

    conn = []
    for j in range(ny):
        for i in range(nx):
            n1 = nid(i, j)
            n2 = nid(i + 1, j)
            n3 = nid(i + 1, j + 1)
            n4 = nid(i, j + 1)
            conn.append([n1, n2, n3, n4])
    return X, np.array(conn, dtype=int)

# -------------------------
# Q4 shape functions
# -------------------------
def q4_shape(xi, eta):
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    ], dtype=float)

    dN_dxi = np.array([
        -0.25 * (1 - eta),
         0.25 * (1 - eta),
         0.25 * (1 + eta),
        -0.25 * (1 + eta),
    ], dtype=float)

    dN_deta = np.array([
        -0.25 * (1 - xi),
        -0.25 * (1 + xi),
         0.25 * (1 + xi),
         0.25 * (1 - xi),
    ], dtype=float)
    return N, dN_dxi, dN_deta

def B_matrix(dN_dx, dN_dy):
    """
    Plane strain kinematics with engineering shear gamma_xy:

      eps_vec = [exx, eyy, gxy]
    """
    nn = len(dN_dx)
    B = np.zeros((3, 2 * nn), dtype=float)
    for a in range(nn):
        B[0, 2*a]     = dN_dx[a]
        B[1, 2*a + 1] = dN_dy[a]
        B[2, 2*a]     = dN_dy[a]
        B[2, 2*a + 1] = dN_dx[a]
    return B

# -------------------------
# Material model: J2 plasticity (local) with linear hardening/softening
# -------------------------
@dataclass
class Material:
    E: float
    nu: float
    sigy0: float   # initial yield stress
    H: float       # isotropic hardening modulus (H<0 for softening)

    @property
    def mu(self):  # shear modulus
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def lam(self): # Lamé lambda
        return (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

@dataclass
class GaussState:
    epsp6: np.ndarray  # (6,) plastic strain tensor components (xx,yy,zz,xy,yz,xz) with xy as tensor component
    epbar: float       # accumulated plastic strain (here: epbar increment == plastic multiplier)

def strain2d_to_6(eps2):
    exx, eyy, gxy = eps2
    exy = 0.5 * gxy
    return np.array([exx, eyy, 0.0, exy, 0.0, 0.0], dtype=float)

def stress6_to_2d(sig6):
    # return [sxx, syy, sxy] with sxy = sigma_xy (tensor shear), compatible with gamma in B
    return np.array([sig6[0], sig6[1], sig6[3]], dtype=float)

def voigt6_to_tensor(sig6):
    sxx, syy, szz, sxy, syz, sxz = sig6
    return np.array([[sxx, sxy, sxz],
                     [sxy, syy, syz],
                     [sxz, syz, szz]], dtype=float)

def tensor_to_voigt6(A):
    return np.array([A[0,0], A[1,1], A[2,2], A[0,1], A[1,2], A[0,2]], dtype=float)

def elastic_stress(mat: Material, eps6, epsp6):
    # σ = 2μ(ε-εp) + λ tr(ε-εp) I
    mu, lam = mat.mu, mat.lam
    eps = voigt6_to_tensor(eps6)
    epsp = voigt6_to_tensor(epsp6)
    ee = eps - epsp
    tr = np.trace(ee)
    sig = 2.0 * mu * ee + lam * tr * np.eye(3)
    return tensor_to_voigt6(sig)

def update_j2_local(mat: Material, eps2: np.ndarray, state_n: GaussState):
    """
    Fully implicit local update at one Gauss point.

    Returns
    -------
    sig2 : (3,) stress in [sxx, syy, sxy] (sxy is tensor shear)
    state_new : GaussState
    """
    mu = mat.mu
    sigy_n = mat.sigy0 + mat.H * state_n.epbar

    eps6 = strain2d_to_6(eps2)
    sig_tr6 = elastic_stress(mat, eps6, state_n.epsp6)
    sig_tr = voigt6_to_tensor(sig_tr6)

    p_tr = np.trace(sig_tr) / 3.0
    s_tr = sig_tr - p_tr * np.eye(3)

    norm_s = np.sqrt(np.sum(s_tr * s_tr))
    if norm_s < 1e-16:
        return stress6_to_2d(sig_tr6), GaussState(state_n.epsp6.copy(), state_n.epbar)

    q_tr = np.sqrt(3.0 / 2.0) * norm_s
    f_tr = q_tr - sigy_n

    if f_tr <= 0.0:
        return stress6_to_2d(sig_tr6), GaussState(state_n.epsp6.copy(), state_n.epbar)

    denom = (3.0 * mu + mat.H)
    if denom <= 1e-12:
        denom = 1e-12

    dlam = f_tr / denom  # == Δepbar
    epbar_new = state_n.epbar + dlam

    q_new = q_tr - 3.0 * mu * dlam
    alpha = q_new / q_tr
    s_new = alpha * s_tr
    sig_new = s_new + p_tr * np.eye(3)

    # plastic strain increment: Δεp_dev = dlam * sqrt(3/2) * n
    n = s_tr / norm_s
    depsp = dlam * np.sqrt(3.0 / 2.0) * n  # deviatoric
    epsp_new = voigt6_to_tensor(state_n.epsp6) + depsp
    state_new = GaussState(tensor_to_voigt6(epsp_new), epbar_new)

    return stress6_to_2d(tensor_to_voigt6(sig_new)), state_new

def consistent_tangent_fd(mat: Material, eps2: np.ndarray, state_n: GaussState):
    """
    Finite-difference tangent: C = dσ/dε (3x3) in (εxx, εyy, γxy).
    Uses the constitutive update (return mapping) for perturbed strains.
    """
    sig0, _ = update_j2_local(mat, eps2, state_n)

    eps_norm = float(np.linalg.norm(eps2))
    h = 1e-8 + 1e-5 * max(1.0, eps_norm)

    C = np.zeros((3, 3), dtype=float)
    for i in range(3):
        de = np.zeros(3, dtype=float)
        de[i] = h
        sig_p, _ = update_j2_local(mat, eps2 + de, state_n)
        sig_m, _ = update_j2_local(mat, eps2 - de, state_n)
        C[:, i] = (sig_p - sig_m) / (2.0 * h)
    return sig0, C

# -------------------------
# FE assembly and solver
# -------------------------
def dof_map(conn_e):
    edofs = np.zeros(2 * len(conn_e), dtype=int)
    for a, n in enumerate(conn_e):
        edofs[2*a] = 2*n
        edofs[2*a + 1] = 2*n + 1
    return edofs

def assemble_system(X, conn, u, states_n, mats_elem):
    ndof = 2 * X.shape[0]
    rows, cols, data = [], [], []
    fint = np.zeros(ndof, dtype=float)

    gp = 1.0 / np.sqrt(3.0)
    gauss = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    w = 1.0

    states_iter = [[None]*4 for _ in range(conn.shape[0])]

    for e, conn_e in enumerate(conn):
        xe = X[conn_e, :]
        ue = u[dof_map(conn_e)]
        Ke = np.zeros((8, 8), dtype=float)
        fe = np.zeros(8, dtype=float)

        mat = mats_elem[e]

        for ig, (xi, eta) in enumerate(gauss):
            _, dN_dxi, dN_deta = q4_shape(xi, eta)
            J = np.zeros((2, 2), dtype=float)
            J[0, 0] = np.dot(dN_dxi, xe[:, 0])
            J[0, 1] = np.dot(dN_dxi, xe[:, 1])
            J[1, 0] = np.dot(dN_deta, xe[:, 0])
            J[1, 1] = np.dot(dN_deta, xe[:, 1])
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError("Non-positive detJ encountered.")
            invJ = np.linalg.inv(J)

            grads = np.vstack((dN_dxi, dN_deta)).T @ invJ.T  # (4,2): [dN/dx, dN/dy]
            dN_dx = grads[:, 0]
            dN_dy = grads[:, 1]

            B = B_matrix(dN_dx, dN_dy)
            eps2 = B @ ue

            state_n = states_n[e][ig]
            sig2, C = consistent_tangent_fd(mat, eps2, state_n)
            _, state_new = update_j2_local(mat, eps2, state_n)
            states_iter[e][ig] = state_new

            Ke += (B.T @ C @ B) * detJ * w
            fe += (B.T @ sig2) * detJ * w

        edofs = dof_map(conn_e)
        fint[edofs] += fe

        rr, cc = np.meshgrid(edofs, edofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(Ke.ravel().tolist())

    K = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K, fint, states_iter

def dirichlet_bcs(X, H, u_top):
    tol = 1e-12
    nnode = X.shape[0]
    fixed = {}

    bottom = np.where(np.abs(X[:, 1] - 0.0) < tol)[0]
    for n in bottom:
        fixed[2*n + 1] = 0.0

    bl = np.argmin((X[:, 0] - 0.0)**2 + (X[:, 1] - 0.0)**2)
    fixed[2*bl + 0] = 0.0

    top = np.where(np.abs(X[:, 1] - H) < tol)[0]
    for n in top:
        fixed[2*n + 1] = u_top

    return fixed, top

def apply_dirichlet(u, fixed):
    for dof, val in fixed.items():
        u[dof] = val
    return u

def solve_nonlinear(X, conn, mats_elem, u_max, nsteps=60, newton_max=25, tol=1e-8, verbose=True):
    nnode = X.shape[0]
    ndof = 2*nnode
    H = float(np.max(X[:, 1]))

    states_n = [[GaussState(epsp6=np.zeros(6), epbar=0.0) for _ in range(4)] for _ in range(conn.shape[0])]
    u = np.zeros(ndof, dtype=float)

    disp_hist = []
    reac_hist = []

    for step in range(1, nsteps + 1):
        u_top = u_max * step / nsteps
        fixed, top_nodes = dirichlet_bcs(X, H=H, u_top=u_top)
        u = apply_dirichlet(u, fixed)

        fixed_dofs = np.array(sorted(fixed.keys()), dtype=int)
        all_dofs = np.arange(ndof, dtype=int)
        free = np.setdiff1d(all_dofs, fixed_dofs)

        converged = False
        for it in range(1, newton_max + 1):
            K, fint, states_iter = assemble_system(X, conn, u, states_n, mats_elem)

            r_free = fint[free]
            norm_r = float(np.linalg.norm(r_free))
            if verbose:
                print(f"[step {step:03d}/{nsteps}] it {it:02d} |R_free| = {norm_r:.3e}")

            if norm_r < tol:
                converged = True
                states_n = states_iter  # commit
                break

            K_ff = K[free][:, free]
            du_free = spla.spsolve(K_ff, -r_free)
            u[free] += du_free
            u = apply_dirichlet(u, fixed)

        if not converged:
            raise RuntimeError(
                f"Newton did not converge at step {step} (u_top={u_top}). "
                f"Try smaller steps, increase newton_max, or use milder softening (H closer to 0)."
            )

        top_uy_dofs = np.array([2*n + 1 for n in top_nodes], dtype=int)
        reaction = -np.sum(fint[top_uy_dofs])
        disp_hist.append(u_top)
        reac_hist.append(reaction)

    return np.array(disp_hist), np.array(reac_hist), u, states_n

def element_epbar(conn, states_n):
    ep = np.zeros(conn.shape[0], dtype=float)
    for e in range(conn.shape[0]):
        ep[e] = np.mean([states_n[e][ig].epbar for ig in range(4)])
    return ep

def plot_elem_field(X, conn, elem_field, title="Element field"):
    cent = np.mean(X[conn], axis=1)
    plt.figure()
    sc = plt.scatter(cent[:, 0], cent[:, 1], c=elem_field, s=25)
    plt.gca().set_aspect("equal", "box")
    plt.colorbar(sc)
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()

def run_mesh_dependency_demo():
    L, H = 1.0, 0.2

    # Pick values that converge robustly (tune if needed)
    mat_base = Material(E=30e9, nu=0.25, sigy0=40e6, H=-5e9)

    weak_band_center = 0.5 * L
    weak_band_halfwidth = 0.03 * L
    weak_factor = 0.9

    u_max = 2e-3
    nsteps = 80

    meshes = [(10, 4), (20, 8), (40, 16)]

    plt.figure()
    for (nx, ny) in meshes:
        X, conn = structured_quad_mesh(nx, ny, L, H)

        cent = np.mean(X[conn], axis=1)
        mats = []
        for c in cent:
            m = mat_base
            if abs(c[0] - weak_band_center) <= weak_band_halfwidth:
                m = Material(E=m.E, nu=m.nu, sigy0=m.sigy0 * weak_factor, H=m.H)
            mats.append(m)

        disp, reac, u, states = solve_nonlinear(
            X, conn, mats,
            u_max=u_max,
            nsteps=nsteps,
            newton_max=30,
            tol=1e-7,
            verbose=False
        )

        plt.plot(disp, reac, label=f"{nx}x{ny} (ne={nx*ny})")

        ep = element_epbar(conn, states)
        plot_elem_field(X, conn, ep, title=f"eq. plastic strain epbar (mesh {nx}x{ny})")

    plt.xlabel("Prescribed top displacement uy [m]")
    plt.ylabel("Reaction force [N]")
    plt.title("Mesh dependency with LOCAL softening plasticity (expected)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_mesh_dependency_demo()
