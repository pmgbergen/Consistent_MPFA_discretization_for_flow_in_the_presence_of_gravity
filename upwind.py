import numpy as np
import scipy
import sympy
import scipy.sparse as sps

import math

import porepy as pp

import xlsxwriter


# Analytical solution
x, y = sympy.symbols('x y')

p0 = 1 #reference pressure

#grid
L = 1
n = 64

domain = np.array([L, L])
grid_dims = np.array([n, n])

pert = 0.5

# Define data
phi = 0.2

rho1 = 100
rho2 = 1000

gx = 0
gy = -10

perm = 1e-12
mu = 1e-3
perm_mu = perm/mu

kd = perm_mu
kc = 0#0.1 * perm_mu

p0 = 1 #reference pressure

ht = domain[1]
hi = domain[1]/2

p = sympy.Piecewise((p0-(ht-y)*rho1*gy, y>=hi), (p0-(ht-hi)*rho1*gy-(hi-y)*rho2*gy, y<hi))
dx = L/n
s = sympy.Piecewise((1, y>=hi), (0, y<hi))
s_eq = sympy.Piecewise((0, y>=hi), (1, y<hi))

l = 1
h1 = domain[1]/4
h2 = domain[1]/4*3
k = 1
k = sympy.Piecewise((1, y<h1), (2, ((y>h1)&(y<hi))), (5, ((y>hi)&(y<h2))), (10, y>h2)) 

perm_xx = k * kd
perm_yy = k * kd
perm_xy = k * kc

perm_xx_f = sympy.lambdify((x, y), perm_xx, 'numpy')
perm_yy_f = sympy.lambdify((x, y), perm_yy, 'numpy')
perm_xy_f = sympy.lambdify((x, y), perm_xy, 'numpy')

p_f = sympy.lambdify((x, y), p, 'numpy')
s_f = sympy.lambdify((x, y), s, 'numpy')
s_eq_f = sympy.lambdify((x, y), s_eq, 'numpy')

dpx = sympy.diff(p, x)
dpy = sympy.diff(p, y)
dpx_f = sympy.lambdify((x, y), dpx, 'numpy')
dpy_f = sympy.lambdify((x, y), dpy, 'numpy')

#rhs = - (sympy.diff(perm_xx * dpx + perm_xy * dpy, x) + sympy.diff(perm_xy * dpx + perm_yy * dpy, y))
#rhs += (sympy.diff(perm_xx * gxa + perm_xy * gya, x) + sympy.diff(perm_xy * gxa + perm_yy * gya, y))
#rhs_f = sympy.lambdify((x, y), rhs, 'numpy')

grid_type = 'cart'

T = 1.0e8

thresh = 1.0e-18

tol_mc = 1.0e-12

def invert_tensor_2d(perm):

    k = np.zeros_like(perm)

    term = perm[0,0,:] * perm[1,1,:] - perm[0,1,:]*perm[1,0,:]

    k[0,0,:] = perm[1,1,:] / term
    k[1,1,:] = perm[0,0,:] / term
    k[1,0,:] = k[0,1,:] = - perm[0,1,:] / term

    return k

def standard_discr(g, k, gforce):

    deviation_from_plane_tol=1e-5

    if g.dim == 2:
        # Rotate the grid into the xy plane and delete third dimension. First
        # make a copy to avoid alterations to the input grid
        g = g.copy()
        cell_centers, face_normals, face_centers, R, _, nodes = pp.map_geometry.map_grid(
            g, deviation_from_plane_tol
        )
        g.cell_centers = cell_centers
        g.face_normals = face_normals
        g.face_centers = face_centers
        g.nodes = nodes

        # Rotate the permeability tensor and delete last dimension
        k = k.copy()
        k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
        k.values = np.delete(k.values, (2), axis=0)
        k.values = np.delete(k.values, (2), axis=1)

    # Step 1

    # take harmonic mean of cell permeability tensors

    fi, ci, sgn = sps.find(g.cell_faces)

    perm = k.values[::, ::, ci]

    # invert cell-centers permeability tensor
    iperm = invert_tensor_2d(perm)
    
    # Distance from face center to cell center
    fc_cc = g.face_centers[::, fi] - g.cell_centers[::, ci]
    dist_face_cell = np.linalg.norm(fc_cc, 2, axis=0)

    # take harmonic mean of permeability k_12 = ((d1 * K1^-1 + d2 * K2^-1)/(d1+d2))^-1
    hperm = np.zeros((2,2,g.num_faces))

    den = np.bincount(fi, weights=dist_face_cell)

    hperm[0,0,:] = np.bincount(fi, weights=dist_face_cell*iperm[0,0,:]) / den
    hperm[1,1,:] = np.bincount(fi, weights=dist_face_cell*iperm[1,1,:]) / den
    hperm[1,0,:] = hperm[0,1,:] = np.bincount(fi, weights=dist_face_cell*iperm[0,1,:]) / den

    hperm = invert_tensor_2d(hperm)

    nk_x = np.sum(g.face_normals[:2]*hperm[0,:,:], axis = 0)
    nk_y = np.sum(g.face_normals[:2]*hperm[1,:,:], axis = 0)

    div_g = np.vstack((nk_x, nk_y))

    # Step 2

    # take arithmetic mean of cell center gravities
    gforce = np.reshape(gforce, (g.num_cells,2))
    gforce = gforce[ci,:]

    fgx = np.bincount(fi, weights=dist_face_cell*gforce[:,0]) / den
    fgy = np.bincount(fi, weights=dist_face_cell*gforce[:,1]) / den
    fg = np.vstack((fgx, fgy))

    flux_g = np.sum(fg[:2]*div_g, axis=0)

    return flux_g

def solve_two_phase_flow_upwind(g):

    time = []
    sat_err = []
    pres_err = []
    maxflux = []
    
    xc = g.cell_centers

    # Permeability tensor
    k_xx = np.zeros(g.num_cells)
    k_xy = np.zeros(g.num_cells)
    k_yy = np.zeros(g.num_cells)

    k_xx[:] = perm_xx_f(xc[0], xc[1])
    k_yy[:] = perm_yy_f(xc[0], xc[1])
    k_xy[:] = perm_xy_f(xc[0], xc[1])

    perm = pp.SecondOrderTensor(k_xx, kyy=k_yy, kxy=k_xy)
    
    # Set type of boundary conditions
    xf = g.face_centers
    p_bound = np.zeros(g.num_faces)

    bnd_faces = g.get_all_boundary_faces()
    neu_faces = bnd_faces[:-1].copy()        
    dir_faces = np.array([g.num_faces-1])

    bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
    top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))
    left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
    right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
   
    bound_cond = pp.BoundaryCondition(g, dir_faces, ['dir'] * dir_faces.size)

    # set value of boundary condition
    p_bound[neu_faces] = 0
    p_bound[dir_faces] = p_f(xf[0, dir_faces], xf[1, dir_faces])

    # set rhs    
    rhs1 = np.zeros(g.num_cells)
    rhs2 = np.zeros(g.num_cells)

    specified_parameters = {"second_order_tensor": perm, "source": rhs1, "bc": bound_cond, "bc_values": p_bound}
    data = pp.initialize_default_data(g, {}, "flow", specified_parameters)
    
    # Set initial conditions
    s = np.zeros(g.num_cells)
    s = s_f(xc[0], xc[1])
    s_0 = s_f(xc[0], xc[1])
    #pp.plot_grid(g, s_0, figsize=(15, 12), alpha = 0.9, color_map=(0,1))
    V_start = np.sum(s*g.cell_volumes)

    fluxtot = np.zeros(g.num_faces)

    p_ex = p_f(xc[0], xc[1])
    s_ex = s_eq_f(xc[0], xc[1])

    time_steps_vec = []

    #save = pp.Exporter(g, 'twophase_gcmpfa8_time', folder='plots')
    #save.write_vtk({"s": s}, 0)

    # Gravity term
    # < K > * rho_alpha * g
    gforce = np.zeros((2, g.num_cells))
    gforce[0,:] = gx * np.ones(g.num_cells)
    gforce[1,:] = gy * np.ones(g.num_cells)
    gforce = gforce.ravel('F')

    g1 = rho1 * gforce
    g2 = rho2 * gforce

    flux_g1 = standard_discr(g, perm, g1)
    flux_g2 = standard_discr(g, perm, g2)
    
    # mpfa discretization
    flux, bound_flux, _, _  = pp.Mpfa("flow")._local_discr(
        g, perm, bound_cond, inverter="python"
    )
    div = pp.fvutils.scalar_divergence(g)

    # define iteration parameters
    t = .0
    k = 0
    i = 0

    time_steps_vec.append(t)
    time_step = 5  

    courant_max = 0.05
    
    while t < T:

        s_old = s.copy()

        # Increment time
        k += 1
        t += time_step

        # discretize gravity term
        kr1, kr2, f2, f1 = get_phase_mobility(g, s, fluxtot, flux_g1, flux_g2)
        fluxtot_g = kr1 * flux_g1 + kr2 * flux_g2
        p_bound[neu_faces] = fluxtot[neu_faces] - fluxtot_g[neu_faces]
        p_bound[bot_faces] *= -1
        p_bound[left_faces] *= -1

        a = div * (kr1 + kr2) * flux
        b = (rhs1 + rhs2) - div * bound_flux * p_bound - div * fluxtot_g

        p = sps.linalg.spsolve(a, b)
        
        fluxtot = (kr1 + kr2) * flux * p + bound_flux * p_bound + fluxtot_g
        assert np.all(abs(fluxtot[neu_faces]) < thresh)
        assert np.all(abs(fluxtot[dir_faces]) < 1.0e-16)

        if pert == 0:
            assert np.all(fluxtot < thresh)

        threshold_indices = abs(fluxtot) < thresh
        fluxtot[threshold_indices] = 0
        
        maxfluxtot = np.amax(np.absolute(fluxtot))
        maxflux.append(maxfluxtot)

        # update s
        q2 = f2 * (fluxtot + kr1 * (flux_g2 - flux_g1))
        q1 = f1 * (fluxtot + kr2 * (flux_g1 - flux_g2))
        q2[bnd_faces]=0
        q1[bnd_faces]=0
        assert np.allclose(q1+q2, fluxtot)

        # check courant number
        q2max = np.zeros(g.num_cells)
        q1max = np.zeros(g.num_cells)
        for j in range (0, g.num_cells):
            faces_of_cell = g.cell_faces.indices[g.cell_faces.indptr[j] : g.cell_faces.indptr[j+1]]
            for i in range (0, faces_of_cell.size):
                q2max[j] += np.absolute(q2[faces_of_cell[i]]) / g.cell_volumes[j]
                q1max[j] += np.absolute(q1[faces_of_cell[i]]) / g.cell_volumes[j]

        courant = 0.5 * max(np.amax(q2max), np.amax(q1max)) * time_step

        div_q2 = div * q2
        threshold_indices = abs(div_q2) < thresh
        div_q2[threshold_indices] = 0

        s = s + time_step / phi / g.cell_volumes * (rhs2 - div_q2)

        if not np.all(s >= 0.0):
            inds=np.argwhere(s<0.0)
            print(k, inds, s[inds])
            assert False
        if not np.all(s <= 1.0):
            inds=np.argwhere(s>1.0)
            #print(k, inds, s[inds])
            assert np.all(s[inds] < 1. + 1.0e-12)
            s[inds] = 1.
            #assert False
            

        mass_error = (np.sum(s*g.cell_volumes) - np.sum(s*g.cell_volumes)) / np.sum(s*g.cell_volumes)

        if abs(mass_error) > tol_mc:
            print('error in mass conservation')
            print(t, mass_error)
            break

        s_diff = s - s_ex
        s_err=np.sqrt(np.sum(g.cell_volumes * s_diff**2))/np.sqrt(np.sum(g.cell_volumes * s_ex**2))

        p_diff = p - p_ex
        p_err=np.sqrt(np.sum(g.cell_volumes * p_diff**2))/np.sqrt(np.sum(g.cell_volumes * p_ex**2))
        
        sat_err.append(s_err)
        pres_err.append(p_err)
        time.append(t)

        #time_steps.append(t)
        #save.write_vtk({"s": s}, time_step=k)

        if k % 100 == 0:
            print(t, mass_error, maxfluxtot, s_err, p_err)
            F = courant_max / courant
            if F > 1.:
                adjust_time_step = min(min(F, 1. + 0.1 * F), 1.2)
                time_step *= adjust_time_step
                print(time_step)

        #time_steps_vec.append(t)
        #i += 1
        """
        if k % 500 == 0 and t < 1.0e8:
            Q_n = fluxtot * g.face_normals
            solver_flow = pp.MVEM("flow")
            P0u = solver_flow.project_flux(g, fluxtot, data)
            #pp.plot_grid(g, p, P0u * 5e7, figsize=(15, 12))
            #pp.plot_grid(g, s_0, figsize=(15, 12), alpha = 0.9, color_map=(0,1))
            pp.plot_grid(g, s, vector_value=Q_n, figsize=(15, 12), vector_scale=1e7, alpha = 0.9, color_map=(0,1))
            #save.write_vtk({"s": s}, time_step=i)


        Q_n = fluxtot * g.face_normals
        pp.plot_grid(g, vector_value=Q_n, figsize=(15, 12), vector_scale=5e7)
        Q_1 = q1 * g.face_normals
        pp.plot_grid(g, vector_value=Q_1, figsize=(15, 12), vector_scale=5e7)
        Q_2 = q2 * g.face_normals
        pp.plot_grid(g, vector_value=Q_2, figsize=(15, 12), vector_scale=5e7)
        """

            
    #save.write_pvd(np.array(time_steps_vec))

    p_diff = p - p_ex
    s_diff = s - s_ex
    p_err=np.sqrt(np.sum(g.cell_volumes * p_diff**2))/np.sqrt(np.sum(g.cell_volumes * p_ex**2))
    s_err=np.sqrt(np.sum(g.cell_volumes * s_diff**2))/np.sqrt(np.sum(g.cell_volumes * s_ex**2))

    mass_error = (np.sum(s*g.cell_volumes) - np.sum(s*g.cell_volumes)) / np.sum(s*g.cell_volumes)
    print('error in mass conservation', mass_error)

    print('error in saturation ', s_err)
    print('error in pressure ', p_err)
    
    pp.plot_grid(g, s, figsize=(15, 12))
    pp.plot_grid(g, p, figsize=(15, 12))

    Q_n = fluxtot * g.face_normals
    solver_flow = pp.MVEM("flow")
    P0u = solver_flow.project_flux(g, fluxtot, data)
    #pp.plot_grid(g, p, P0u * 5e7, figsize=(15, 12))
    #pp.plot_grid(g, s_0, figsize=(15, 12), alpha = 0.9, color_map=(0,1))
    #pp.plot_grid(g, s, vector_value=Q_n, figsize=(15, 12), vector_scale=1e7, alpha = 0.9, color_map=(0,1))
    #save.write_vtk({"s": s}, time_step=i)

    #save.write_vtk({"p": p, 's_0' : s_0, 's_f' : s})

    return time, sat_err, pres_err, maxflux

def get_hybrid_upwind(g, s, q):

    kr1 = np.zeros_like(q)
    kr2 = np.zeros_like(q)
    kr1_g = np.zeros_like(q)
    kr2_g = np.zeros_like(q)
    
    bf = g.get_all_boundary_faces()
    
    fi, ci, sgn = scipy.sparse.find(g.cell_faces)

    for j in range (0, g.num_faces):

        if np.isin(j, bf):           
            # for boundary faces, take corresponding cell
            k = np.ravel(np.argwhere(fi == j))
            kr1[j] = k1(s[ci[k]])
            kr2[j] = k2(s[ci[k]])
        else:
            # for internal faces, evaluate flux
            i = np.ravel(np.argwhere((fi == j)
                          & (sgn>0)))
            ip = np.ravel(np.argwhere((fi == j)
                                      & (sgn<0)))

            if q[j] >= 0:
                kr2[j] = k2(s[ci[i]])
                kr1[j] = k1(s[ci[i]])
            else:
                kr2[j] = k2(s[ci[ip]])
                kr1[j] = k1(s[ci[ip]])

            kr2_g[j] = k2(s[ci[ip]])
            kr1_g[j] = k1(s[ci[i]])

    # construct diagonal matrix
    faces = np.arange(g.num_faces)
    kr1_mat = sps.coo_matrix((kr1, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()
    kr2_mat = sps.coo_matrix((kr2, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()
    kr1_gmat = sps.coo_matrix((kr1_g, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()
    kr2_gmat = sps.coo_matrix((kr2_g, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()
    f2mat = sps.coo_matrix((kr2/(kr1+kr2+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()

    f1mat = sps.coo_matrix((kr1/(kr1+kr2+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()  

    f2_gmat = sps.coo_matrix((kr2_g/(kr1_g+kr2_g+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()
    f1_gmat = sps.coo_matrix((kr1_g/(kr1_g+kr2_g+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()

   
    return f1mat, f2mat, kr1_mat, kr2_mat, f1_gmat, f2_gmat, kr1_gmat, kr2_gmat

def get_gravity_mobilities(g, s):

    kr1 = np.zeros(g.num_faces)
    kr2 = np.zeros(g.num_faces)
    
    bf = g.get_all_boundary_faces()
    
    fi, ci, sgn = scipy.sparse.find(g.cell_faces)

    for j in range (0, g.num_faces):

        if np.isin(j, bf):           
            # for boundary faces, take corresponding cell
            k = np.ravel(np.argwhere(fi == j))
            kr1[j] = k1(s[ci[k]])
            kr2[j] = k2(s[ci[k]])
        else:
            # for internal faces, evaluate flux
            i = np.ravel(np.argwhere((fi == j)
                          & (sgn>0)))
            ip = np.ravel(np.argwhere((fi == j)
                                      & (sgn<0)))

            kr2[j] = k2(s[ci[ip]])
            kr1[j] = k1(s[ci[i]])

    # construct diagonal matrix
    faces = np.arange(g.num_faces)

    kr1_mat = sps.coo_matrix((kr1, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()

    kr2_mat = sps.coo_matrix((kr2, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()
    
    return kr1_mat, kr2_mat

def get_phase_mobility(g, s, q, kg1, kg2):

    kr1 = np.zeros_like(q)
    kr2 = np.zeros_like(q)

    bf = g.get_all_boundary_faces()
    
    fi, ci, sgn = scipy.sparse.find(g.cell_faces)

    for j in range (0, g.num_faces):

        if np.isin(j, bf):           
            # for boundary faces, take corresponding cell
            k = np.ravel(np.argwhere(fi == j))
            kr1[j] = k1(s[ci[k]])
            kr2[j] = k2(s[ci[k]])
        else:
            # for internal faces, evaluate flux
            i = np.ravel(np.argwhere((fi == j)
                          & (sgn>0)))
            ip = np.ravel(np.argwhere((fi == j)
                                      & (sgn<0)))

            if q[j] == 0:
                if (kg2[j]-kg1[j]) > 0:
                    kr2[j] = k2(s[ci[i]])
                    kr1[j] = k1(s[ci[ip]])
                else:
                    kr2[j] = k2(s[ci[ip]])
                    kr1[j] = k1(s[ci[i]])
            else:
                if q[j] >= 0:
                    upw = i
                    opp = ip
                else:
                    upw = ip
                    opp = i

                if q[j] * kg1[j] > 0:
                    kr2[j] = k2(s[ci[upw]])
                    kr1[j] = k1(s[ci[upw]])
                    q1 = kr1[j] / (kr1[j] + kr2[j]) * (q[j] + kr2[j] * (kg1[j]-kg2[j]))
                    if np.sign(q1) == np.sign(q[j]):
                        pass
                    else:
                        kr1[j] = k1(s[ci[opp]])
                else:
                    kr1[j] = k1(s[ci[upw]])
                    kr2[j] = k2(s[ci[upw]])
                    q2 = kr2[j] / (kr1[j] + kr2[j]) * (q[j] + kr1[j] * (kg2[j]-kg1[j]))
                    if np.sign(q2) == np.sign(q[j]):
                        pass
                    else:
                        kr2[j] = k2(s[ci[opp]])          

    # construct diagonal matrix
    faces = np.arange(g.num_faces)

    kr1mat = sps.coo_matrix((kr1, (faces, faces)),
                               shape=(g.num_faces,
                                      g.num_faces)).tocsr()

    kr2mat = sps.coo_matrix((kr2, (faces, faces)),
                           shape=(g.num_faces,
                                  g.num_faces)).tocsr()

    f2mat = sps.coo_matrix((kr2/(kr1+kr2+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()  

    f1mat = sps.coo_matrix((kr1/(kr1+kr2+thresh), (faces, faces)),
                       shape=(g.num_faces,
                              g.num_faces)).tocsr()
    
    return kr1mat, kr2mat, f2mat, f1mat           
    
def k1(s):
    return (1 - s)**2
def k2(s):
    return s**2

def perturb(g, rate, dx, alpha):
    rand = np.vstack((np.random.rand(g.dim, g.num_nodes), np.repeat(0., g.num_nodes)))
    r1 = np.ravel(np.argwhere((g.nodes[0] < domain[0] - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < hi - alpha) & (g.nodes[1] > 1e-10)))
    r2 = np.ravel(np.argwhere((g.nodes[0] < domain[0] - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < ht - 1e-10) & (g.nodes[1] > hi + alpha)))
    #r3 = np.ravel(np.argwhere((g.nodes[0] < 1 - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < 0.75 - 1e-10) & (g.nodes[1] > 0.5 + 1e-10)))
    #r4 = np.ravel(np.argwhere((g.nodes[0] < 1 - 1e-10) & (g.nodes[0] > 1e-10) & (g.nodes[1] < 1.0 - 1e-10) & (g.nodes[1] > 0.75 + 1e-10)))
    pert_nodes = np.concatenate((r1, r2))
    npertnodes = pert_nodes.size
    rand = np.vstack((np.random.rand(g.dim, npertnodes), np.repeat(0., npertnodes)))
    g.nodes[:,pert_nodes] += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0

    np.random.seed(42)
    return g

g = pp.CartGrid(grid_dims, domain)

dx = np.max(domain / grid_dims)
alpha = 0.5 * dx

if pert != 0:
    g = perturb(g, pert, dx, alpha)

g.compute_geometry()

t, s, p, q = solve_two_phase_flow_upwind(g)

# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('upwind_hperm_n64.xlsx')
worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0

array = np.array([t, s, p, q])
# Iterate over the data and write it out row by row.

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)
workbook.close()

