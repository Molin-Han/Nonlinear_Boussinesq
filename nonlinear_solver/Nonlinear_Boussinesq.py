from firedrake import *
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from firedrake.output import VTKFile
from petsc4py import PETSc

print = PETSc.Sys.Print
class NonlinearBoussinesq:
    def __init__(self, N=1.0e-2, U=0., dt=600., nx=5e3, ny=1, Lx=1e3, Ly=1, height=1e4, nlayers=20):

        # Extruded Mesh 3D
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.height = height
        self.U = U # steady velocity
        self.N = N # buoyancy frequency
        self.ar = height / Lx
        self.dx = Lx / nx
        self.dz = height / nlayers

        # Create the mesh
        self.m = PeriodicRectangleMesh(self.nx, self.ny, self.Lx, self.Ly, direction='both',quadrilateral=True)
        # Build the mesh hierarchy for the extruded mesh to construct vertically constant spaces.
        self.mh = MeshHierarchy(self.m, refinement_levels=0)
        self.hierarchy = ExtrudedMeshHierarchy(self.mh, height,layers=[1, nlayers], extrusion_type='uniform')
        self.mesh = ExtrudedMesh(self.m, nlayers, layer_height = height/nlayers, extrusion_type='uniform')

        # Mixed Finite Element Space
        horizontal_degree = 2
        vertical_degree = 2

        # horizontal base spaces -- 2D
        S1 = FiniteElement("RTCF", quadrilateral, horizontal_degree) # RT2 in 2D
        S2 = FiniteElement("DQ", quadrilateral, horizontal_degree-1) # DG1 in 2D
        # vertical base spaces
        T0 = FiniteElement("CG", interval, vertical_degree) # CG2
        T1 = FiniteElement("DG", interval, vertical_degree-1) # DG1

        # Attempt to build the 3D element.
        Vh_elt = TensorProductElement(S1, T1)
        Vh = HDivElement(Vh_elt)
        Vv_elt = TensorProductElement(S2, T0)
        Vv = HDivElement(Vv_elt)
        V_3d = Vh + Vv

        V = FunctionSpace(self.mesh, V_3d, name="HDiv") # Velocity space RT(k-1)
        Vb = FunctionSpace(self.mesh, Vv_elt, name="Buoyancy") # Buoyancy space
        Vp_elt = TensorProductElement(S2, T1) # DG horizontal and DG vertical
        Vp = FunctionSpace(self.mesh, Vp_elt, name="Pressure") # Pressure space

        self.W = V * Vb * Vp # velocity, buoyancy, pressure space
        self.x, self.y, self.z = SpatialCoordinate(self.mesh)

        # Setting up the solution variables.
        self.Un = Function(self.W)
        self.Unp1 = Function(self.W)
        self.un, self.bn, self.pn = split(self.Un)
        self.unp1, self.bnp1, self.pnp1 = split(self.Unp1)
        self.w, self.q, self.phi = TestFunctions(self.W)

        # Setting up the intermediate variables for second order accuracy.
        self.unph = 0.5 * (self.un + self.unp1)
        self.bnph = 0.5 * (self.bn + self.bnp1)
        self.pnph = 0.5 * (self.pn + self.pnp1)

        # Setting up the normal vector, buoyancy frequency, coriolis parameter and time step.
        self.n = FacetNormal(self.mesh) # outward normal vector
        self.unn = 0.5*(dot(self.un, self.n) + abs(dot(self.un, self.n))) # upwinding variable.
        self.k = as_vector([0, 0, 1])
        # Setting up the Coriolis parameter
        Omega = 7.292e-5
        theta = pi / 3
        self.omega = as_vector([0, Omega * sin(theta), Omega * cos(theta)])
        self.dt = Constant(dt)

    def build_initial_data(self):
        xc = self.Lx/2
        yc = self.Ly/2
        a = Constant(5000)
        U = Constant(self.U)
        un, bn, pn = self.Un.subfunctions
        unp1, bnp1, pnp1 = self.Unp1.subfunctions
        un.project(as_vector([U,Constant(0.0),Constant(0.0)]))
        unp1.project(as_vector([U,Constant(0.0),Constant(0.0)]))
        # Solve the equation for the whole variable instead of only the perturbed variable.
        bn.project(sin(pi*self.z/self.height)/(1+((self.x-xc)**2+(self.y-yc)**2)/a**2) + self.N**2 * self.z)
        bnp1.project(sin(pi*self.z/self.height)/(1+((self.x-xc)**2+(self.y-yc)**2)/a**2) + self.N**2 * self.z)

        # Project the hydrostatic pressure as initial guess.
        DG = FunctionSpace(self.mesh, 'DG', 0)
        One = Function(DG).assign(1.0)
        area = assemble(One*dx)
        pn.project(0.5 * self.N**2 * self.z**2)
        pnp1.project(0.5 * self.N**2 * self.z**2)
        pn_int = assemble(pn*dx)
        pn.project(pn - pn_int/area)
        pnp1_int = assemble(pnp1*dx)
        pnp1.project(pnp1 - pnp1_int/area)
        print("Calulated hydrostatic pressure as initial guess and satisfies the pressure condition.")


    def build_lu_params(self):
        self.params = {'ksp_type': 'preonly', 'pc_type':'lu', 'mat_type': 'aij', 'pc_factor_mat_solver_type': 'mumps'}
    
    def build_ASM_MH_params(self):
        self.params = {
            'mat_type': 'matfree',
            'ksp_type': 'gmres',
            'snes_monitor': None,
            # 'snes_type':'ksponly',
            'ksp_monitor': None,
            # "ksp_monitor_true_residual": None,
            'pc_type': 'mg',
            'pc_mg_type': 'full',
            "ksp_converged_reason": None,
            "snes_converged_reason": None,
            'mg_levels': {
                'ksp_type': 'richardson',
                # "ksp_monitor_true_residual": None,
                # "ksp_view": None,
                'ksp_max_it': 1,
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled_pc_type': 'python',
                'assembled_pc_python_type': 'firedrake.ASMVankaPC',
                'assembled_pc_vanka_construct_dim': 0,
                'assembled_pc_vanka_sub_sub_pc_type': 'lu'
                #'assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type':'mumps'
                },
            'mg_coarse': {
                'ksp_type': 'preonly',
                'pc_type': 'lu'
                }
            }


    def build_pure_Vanka_params(self):
        self.params = {
            "mat_type": "matfree",
            "ksp_type": "gmres",
            'snes_monitor': None,
            # "snes_view": None,
            "snes_converged_reason": None,
            "ksp_converged_reason": None,
            'ksp_monitor': None,
            # "ksp_monitor_true_residual": None,
            # "ksp_view": None,
            "ksp_atol": 1e-8,
            "ksp_rtol": 1e-8,
            "ksp_max_it": 400,
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "python",
            "assembled_pc_python_type": "firedrake.ASMVankaPC",
            "assembled_pc_vanka_construct_dim": 0,
            "assembled_pc_vanka_sub_sub_pc_type": "lu",
            # "assembled_pc_vanka_sub_sub_pc_factor_mat_solver_type":'mumps',
            # "assembled_pc_vanka_sub_sub_pc_factor_mat_ordering_type":'rcm',
            }


    def build_boundary_condition(self):
        # Boundary conditions
        bc1 = DirichletBC(self.W.sub(0), as_vector([0., 0., 0.]), "top")
        bc2 = DirichletBC(self.W.sub(0), as_vector([0., 0., 0.]), "bottom")
        bc3 = DirichletBC(self.W.sub(0), as_vector([0., 0., 0.]), "on_boundary")
        self.bcs = [bc1, bc2, bc3]

    def build_NonlinearVariationalSolver(self):
        # Simplify variable name
        un, pn, bn = self.un, self.pn, self.bn
        unp1, pnp1, bnp1 = self.unp1, self.pnp1, self.bnp1
        unph, pnph, bnph = self.unph, self.pnph, self.bnph
        w, phi, q = self.w, self.phi, self.q
        unn = self.unn
        k = self.k
        dt = self.dt
        N = self.N
        omega = self.omega
        Ubar = as_vector([Constant(self.U), 0, 0])
        def u_eqn(w):
            return (
                inner(w, (unp1 - un)) * dx +
                dt * inner(w, 2 * cross(omega, unph)) * dx -
                dt * div(w) * pnph * dx - dt * inner(w, k) * bnph * dx
            )

        def b_eqn(q):
            return (
                q * (bnp1 - bn) * dx +
                dt * N**2 * q * inner(k, unph) * dx -
                dt * div(q * Ubar) * bnph * dx +
                dt * jump(q) * (unn('+') * bnph('+') - unn('-') * bnph('-')) * (dS_v + dS_h)
            )

        def p_eqn(phi):
            return (
                phi * div(unph) * dx
            )

        eqn = u_eqn(w) + b_eqn(q) + p_eqn(phi)
        bcs = self.bcs
        self.nprob = NonlinearVariationalProblem(eqn, self.Unp1, bcs=bcs)

        # Nullspace for the problem
        v_basis = VectorSpaceBasis(constant=True,comm=COMM_WORLD) #pressure field nullspace
        self.nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), self.W.sub(1), v_basis])
        trans_null = VectorSpaceBasis(constant=True,comm=COMM_WORLD)
        self.trans_nullspace = MixedVectorSpaceBasis(self.W, [self.W.sub(0), self.W.sub(1), trans_null])
        self.nsolver = NonlinearVariationalSolver(
                                                    self.nprob,
                                                    nullspace=self.nullspace,
                                                    transpose_nullspace=self.trans_nullspace,
                                                    solver_parameters=self.params,
                                                    options_prefix='linear_boussinesq_ASM'
                                                    )

    def time_stepping(self, tmax=3600.0, dt=600.0, monitor=False, xtest=False, ztest=False, artest=False):
        Un = self.Un
        Unp1 = self.Unp1

        name = "lb_imp"
        file_lb = VTKFile(name+'.pvd')
        un, bn, Pin = Un.subfunctions
        file_lb.write(un, Pin, bn)
        Unp1.assign(Un)

        t = 0.0
        dumpt = 600.
        tdump = 0.
        self.dt.assign(dt)
        print('tmax=', tmax, 'dt=', self.dt)
        while t < tmax - 0.5*dt:
            print(t)
            t += dt
            tdump += dt
            if monitor:
                self.sol_final = np.loadtxt(f'sol_final_{int(t)}.out')
                error_list = []
                # Set a monitor
                def my_monitor_func(ksp, iteration_number, norm):
                    #print(f"The monitor is operating with current iteration {iteration_number}")
                    sol = ksp.buildSolution()
                    # Used relative error here
                    err = np.linalg.norm(self.sol_final - sol.getArray(), ord=2) / np.linalg.norm(self.sol_final)
                    #print(f"error norm is {err}")
                    error_list.append(err)
                self.nsolver.snes.ksp.setMonitor(my_monitor_func)
                self.nsolver.solve()
                # print(error_list)
                print("Monitor is on and working.")
                if artest:
                    # test for the aspect ratio
                    np.savetxt(f'err_ar_{self.ar}_{int(t)}.out', error_list)
                if xtest:
                    # test for the different dx
                    np.savetxt(f'err_dx_{self.dx}_{int(t)}.out', error_list)
                if ztest:
                    # test for the different dz
                    np.savetxt(f'err_dz_{self.dz}_{int(t)}.out', error_list)
            else:
                self.nsolver.solve()
                self.sol_final = self.nsolver.snes.ksp.getSolution().getArray()
                np.savetxt(f'sol_final_{int(t)}.out',self.sol_final)
                print("The nonlinear solver is solved and final solution is saved.")
            self.Un.assign(self.Unp1)

            if tdump > dumpt - dt*0.5:
                file_lb.write(un, Pin, bn)
                tdump -= dumpt


if __name__ == "__main__":
    N=1.0e-2
    U=0.
    dt=600.0
    tmax = 6000.0
    nx=30
    ny=1
    Lx=3.0e5
    Ly=1.0e-3 * Lx
    height=1e4
    nlayers=10

    eqn = NonlinearBoussinesq(N=N, U=U, dt=dt, nx=nx, ny=ny, Lx=Lx, Ly=Ly, height=height, nlayers=nlayers)
    eqn.build_initial_data()
    # eqn.build_lu_params()
    eqn.build_ASM_MH_params()
    # eqn.build_pure_Vanka_params()
    eqn.build_boundary_condition()
    eqn.build_NonlinearVariationalSolver()
    eqn.time_stepping(tmax=tmax, dt=dt, monitor=True)
    print("The simulation is completed.")
