import numpy as np
import scipy
import sympy

import scipy.sparse as sps

import math

import porepy as pp

###
np.random.seed(42)
base = 4
domain = np.array([1, 1])
basedim = np.array([base, base])
num_refs = 7
ref_rate=2
### End of parameter definitions

xr = domain[0]
x0 = xr/2
yt = domain[1]
y0 = yt/2

# Permeability tensor
kd = 1
kc = 0.1

# Analytical solution
x, y = sympy.symbols('x y')

# Discontinuity line: rx + sy = d
r = 0.7

s = 1 - r
# passing to the central point
d = r * x0 + s * y0

theta = math.pi/2 - math.atan(-r/s)

# jump in permeability (1 if no jump)
l = 1

f = r * x + s * y

g1 = 10#sympy.sin(x) * sympy.cos(y) 
g2 = 1# * x - 3 * y

n = math.sqrt(r*r+s*s)

d1 = - (f - d) / n
d2 = (f - d) / n

u0 = 10

test = 1

if test == 1:
    a1 = 1
    a2 = 0
if test == 2:
    a1 = 0
    a2 = 1
if test == 3:
    a1 = 1
    a2 = 1
if test == 4:
    a1 = 1
    a2 = 100
u = u0 + a1 * sympy.Piecewise((g1 * d1, ((f>=0)&(f<d))), ((- g2 * d2, ((f>=d)&(f<=1))))) + a2 * sympy.sin(x) * sympy.cos(y)
gx = sympy.diff(u, x)
gy = sympy.diff(u, y)

k = sympy.Piecewise((1, ((f>=0)&(f<d))), (l, ((f>=d)&(f<=1)))) 

perm_xx = kd * k
perm_yy = kd * k
perm_xy = kc * k

perm_xx_f = sympy.lambdify((x, y), perm_xx, 'numpy')
perm_yy_f = sympy.lambdify((x, y), perm_yy, 'numpy')
perm_xy_f = sympy.lambdify((x, y), perm_xy, 'numpy')

u_f = sympy.lambdify((x, y), u, 'numpy')
gx_f = sympy.lambdify((x, y), gx, 'numpy')
gy_f = sympy.lambdify((x, y), gy, 'numpy')

dux = sympy.diff(u, x)
duy = sympy.diff(u, y)
dux_f = sympy.lambdify((x, y), dux, 'numpy')
duy_f = sympy.lambdify((x, y), duy, 'numpy')

rhs = - (sympy.diff(perm_xx * dux + perm_xy * duy, x) + sympy.diff(perm_xy * dux + perm_yy * duy, y))
rhs += (sympy.diff(perm_xx * gx + perm_xy * gy, x) + sympy.diff(perm_xy * gx + perm_yy * gy, y))
rhs_f = sympy.lambdify((x, y), rhs, 'numpy')

bctype = 'neu'

eps = 1.0e-8
deviation_from_plane_tol=1e-5

def invert_tensor_2d(perm):

    k = np.zeros_like(perm)

    term = perm[0,0,:] * perm[1,1,:] - perm[0,1,:]*perm[1,0,:]

    k[0,0,:] = perm[1,1,:] / term
    k[1,1,:] = perm[0,0,:] / term
    k[1,0,:] = k[0,1,:] = - perm[0,1,:] / term

    return k

def standard_discr(g, k, gforce):

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

def run_convergence(grid_type, gravity):
    u_err = []
    flux_err = []
    hs = []

    for iter1 in range(num_refs):

        dim = basedim.shape[0]

        grid_dims = basedim * ref_rate ** iter1

        g = pp.CartGrid(grid_dims, domain)
        g.compute_geometry()

        if r != 1.:        
            g = rotate(g, theta, domain, grid_dims)
     
        xc = g.cell_centers

        # Permeability tensor
        k = pp.SecondOrderTensor(np.ones(g.num_cells))

        k_xx = np.zeros(g.num_cells)
        k_xy = np.zeros(g.num_cells)
        k_yy = np.zeros(g.num_cells)

        k_xx[:] = perm_xx_f(xc[0], xc[1])
        k_yy[:] = perm_yy_f(xc[0], xc[1])
        k_xy[:] = perm_xy_f(xc[0], xc[1])

        k = pp.SecondOrderTensor(k_xx, kyy=k_yy, kxy=k_xy)

        # Gravity
        gforce = np.zeros((2, g.num_cells))
        gforce[0,:] = gx_f(xc[0], xc[1])
        gforce[1,:] = gy_f(xc[0], xc[1])
        gforce = gforce.ravel('F')

        # Set type of boundary conditions
        xf = g.face_centers
        u_bound = np.zeros(g.num_faces)

        if bctype == 'dir':
            dir_faces = g.get_all_boundary_faces()
        else:
            # Dir left and right
            left_faces = np.ravel(np.argwhere(g.face_centers[0] < 1e-10))
            right_faces = np.ravel(np.argwhere(g.face_centers[0] > domain[0] - 1e-10))
            # Neu bottom and top
            bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
            top_faces = np.ravel(np.argwhere(g.face_centers[1] > domain[1] - 1e-10))

            dir_faces = np.concatenate((left_faces, right_faces))
            neu_faces = np.concatenate((bot_faces, top_faces))

        bound_cond = pp.BoundaryCondition(g, dir_faces, ['dir'] * dir_faces.size)

        # set value of boundary condition
        u_bound[dir_faces] = u_f(xf[0, dir_faces], xf[1, dir_faces])
 
        # Exact solution
        u_ex = u_f(xc[0], xc[1])

        kgradpx = perm_xx_f(xf[0], xf[1])*dux_f(xf[0], xf[1])+perm_xy_f(xf[0], xf[1])*duy_f(xf[0], xf[1])
        kgradpy = perm_xy_f(xf[0], xf[1])*dux_f(xf[0], xf[1])+perm_yy_f(xf[0], xf[1])*duy_f(xf[0], xf[1])
        du_ex_faces = np.vstack((kgradpx, kgradpy))

        kgx = perm_xx_f(xf[0], xf[1])*gx_f(xf[0], xf[1]) + perm_xy_f(xf[0], xf[1]) * gy_f(xf[0], xf[1])
        kgy = perm_xy_f(xf[0], xf[1])*gx_f(xf[0], xf[1]) + perm_yy_f(xf[0], xf[1]) * gy_f(xf[0], xf[1]) 
        g_ex_faces = np.vstack((kgx, kgy))

        flux_ex_du = -np.sum(g.face_normals[:2] * du_ex_faces, axis=0)
        flux_ex_g = np.sum(g.face_normals[:2] * g_ex_faces, axis=0)
        flux_ex = flux_ex_du + flux_ex_g 

        # MPFA discretization, and system matrix
        if gravity:
            flux, bound_flux, _, _, div_g  = pp.Mpfa("flow").mpfa(
                g, k, bound_cond, vector_source=gravity, inverter="python"
            )
            flux_g = div_g * gforce
        else:
            flux, bound_flux, _, _  = pp.Mpfa("flow").mpfa(
                g, k, bound_cond, inverter="python"
            )
            flux_g = standard_discr(g, k, gforce)

        div = pp.fvutils.scalar_divergence(g)
        a = div * flux
            
        if bctype == 'neu':
            if gravity == False:
                u_bound[neu_faces] = flux_ex[neu_faces] - flux_ex_g[neu_faces]
                flux_g[neu_faces] = flux_ex_g[neu_faces]
                if grid_type == 'cart':
                    u_bound[bot_faces] *= -1
            else:
                u_bound[neu_faces] = flux_ex[neu_faces]
                u_bound[neu_faces] = 0
      
        # Right hand side - contribution from the solution and the boundary conditions
        xc = g.cell_centers
        rhs = rhs_f(xc[0], xc[1]) * g.cell_volumes
        b = rhs - div * bound_flux * u_bound - div * flux_g
       
        # Solve system, derive fluxes
        u_num = scipy.sparse.linalg.spsolve(a, b)
        #pp.plot_grid(g, u_num, figsize=(15, 12))
        #save = pp.Exporter(g, 'thetagrid', folder='plots')
        #cell_id = np.arange(g.num_cells)
        #save.write_vtk({"pressure": u_num})

        flux_num_du = flux * u_num + bound_flux * u_bound
        flux_num = flux_num_du + flux_g

        assert np.all(abs(flux_num[neu_faces]) < 1.0e-12)

        # Calculate errors
        u_diff = u_num - u_ex
        flux_diff = flux_num - flux_ex

        hs.append(g.num_cells**(-1/g.dim))
        u_err.append(np.sqrt(np.sum(g.cell_volumes * u_diff**2))/np.sqrt(np.sum(g.cell_volumes * u_ex**2)))
        den = np.sqrt(np.sum((g.face_areas ** g.dim) * flux_ex**2))
        num = np.sqrt(np.sum((g.face_areas ** g.dim) * flux_diff**2))
        if den != 0:
            flux_err.append(num/den)
        else:
            flux_err.append(num)

    return u_err, flux_err, hs

def perturb(g, rate, dx):
    rand = np.vstack((np.random.rand(g.dim, g.num_nodes), np.repeat(0., g.num_nodes)))
    r1 = np.ravel(np.argwhere((g.nodes[0] > 1e-10) & (g.nodes[1] > 1e-10) & (g.nodes[1] < 1.0 - 1e-10) & (r*g.nodes[0]+s*g.nodes[1] < -1e-10)))
    r2 = np.ravel(np.argwhere((g.nodes[0] < 1 - 1e-10) & (g.nodes[1] > 1e-10) & (g.nodes[1] < 1.0 - 1e-10) & (r*g.nodes[0]+s*g.nodes[1] > 1e-10)))
    pert_nodes = np.concatenate((r1, r2))
    npertnodes = pert_nodes.size
    rand = np.vstack((np.random.rand(g.dim, npertnodes), np.repeat(0., npertnodes)))
    g.nodes[:,pert_nodes] += rate * dx * (rand - 0.5)
    # Ensure there are no perturbations in the z-coordinate
    if g.dim == 2:
        g.nodes[2, :] = 0
    return g

def rotate(g, theta, domain, nx):

    bot_left_nodes = np.ravel(np.argwhere((g.nodes[0] > 1.0e-10) & (g.nodes[0] < 0.5 * domain[0] + 1.0e-10) & (g.nodes[1] < 0.5 * domain[1])))
    bot_right_nodes = np.ravel(np.argwhere((g.nodes[0] > 0.5 * domain[0]) & (g.nodes[0] < domain[0]) & (g.nodes[1] < 0.5 * domain[1])))
    top_left_nodes = np.ravel(np.argwhere((g.nodes[0] > 1.0e-10) & (g.nodes[0] < 0.5 * domain[0] + 1.0e-10) & (g.nodes[1] > 0.5 * domain[1])))
    top_right_nodes = np.ravel(np.argwhere((g.nodes[0] > 0.5 * domain[0]) & (g.nodes[0] < domain[0] - 1.0e-10) & (g.nodes[1] > 0.5 * domain[1])))

    incr1 = []
    incr2 = []
    incr3 = []
    incr4 = []

    for j in range (0, int(nx[1])//2):

        h1 = g.nodes[1, (nx[0]+1)*j]
        h2 = domain[1]/nx[1] * (1 + j)

        l1 = 0.5 * domain[0] - (0.5 * domain[1] - h1) * math.tan(theta)
        l2 = 0.5 * domain[0] + h2 * math.tan(theta)

        l1_r = domain[0] - l1
        l2_r = domain[0] - l2

        dx1 = 2 * l1 / nx[0]
        dx2 = 2 * l2 / nx[0]

        dx1_r = 2 * l1_r / nx[0]
        dx2_r = 2 * l2_r / nx[0]

        for i in range (0, int(nx[0]/2)):
            incr1.append((i+1)*dx1)
            incr2.append((i+1)*dx2)

            if i > 0:
                incr3.append(l1 + i*dx1_r)
                incr4.append(l2 + i*dx2_r)


    g.nodes[0, bot_left_nodes] = incr1
    g.nodes[0, top_left_nodes] = incr2
    g.nodes[0, bot_right_nodes] = incr3
    g.nodes[0, top_right_nodes] = incr4

    g.compute_geometry()
    
    return g
    
grids = ['cart']

for gr in grids:
    eu, eq, hs = run_convergence(gr, gravity = True)

    print('GCMPFA')

    print('The error in displacement: ', eu)
    print('The error in flux: ', eq)

    h_vec = np.array(hs)
    u_vec = np.array(eu)
    q_vec = np.array(eq)

    u_rate = (np.log(u_vec[1:] / u_vec[:-1])/
                  np.log(h_vec[1:] / h_vec[:-1]))
    q_rate = (np.log(q_vec[1:] / q_vec[:-1])/
                  np.log(h_vec[1:] / h_vec[:-1]))

    print('The convergence rate for displacement: ', u_rate)
    print('The convergence rate for flux: ', q_rate)

    eu, eq, hs = run_convergence(gr, gravity = False)

    print('STANDARD')

    print('The error in displacement: ', eu)
    print('The error in flux: ', eq)

    h_vec = np.array(hs)
    u_vec = np.array(eu)
    q_vec = np.array(eq)

    u_rate = (np.log(u_vec[1:] / u_vec[:-1])/
                  np.log(h_vec[1:] / h_vec[:-1]))
    q_rate = (np.log(q_vec[1:] / q_vec[:-1])/
                  np.log(h_vec[1:] / h_vec[:-1]))

    print('The convergence rate for displacement: ', u_rate)
    print('The convergence rate for flux: ', q_rate)
