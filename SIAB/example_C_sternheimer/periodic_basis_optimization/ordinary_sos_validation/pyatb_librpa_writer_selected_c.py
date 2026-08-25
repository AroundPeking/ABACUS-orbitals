import numpy as np
import sys
def output_librpa(lattice_vector: np.array, fermi_energy: float, occ_band: int, nkx : int = 20, nky : int = 20, nkz : int = 20, nspin: int = 1, matrix_route: str = 'OUT.ABACUS', use_soc: bool = False):
    import pyatb
    from pyatb import RANK, COMM, SIZE
    from pyatb.kpt.kpoint_generator import mp_generator, kpoints_in_different_process
    from pyatb.parallel import op_sum
    from pyatb.tools.smearing import gauss
    import os

    """----------------输入数据----------------"""
    # 1. 晶格参数
    lattice_constant = 1.0
    # unit: \AA
    #lattice_vector = np.array(
    #    [
    #        [0.000000000000,  1.8,  1.8],
    #        [1.8,  0.000000000000,  1.8],
    #        [1.8,  1.8,  0.000000000000]
    #    ], dtype=float
    #)
    if(nspin==2):
        HR_route = [os.path.join(matrix_route, 'hrs1_nao.csr'), os.path.join(matrix_route, 'hrs2_nao.csr')]
    if(nspin==1 or nspin==4):
        HR_route = os.path.join(matrix_route, 'hrs1_nao.csr')
    SR_route = os.path.join(matrix_route, 'srs1_nao.csr')
    rR_route = os.path.join(matrix_route, 'rr.csr')
    pR_route = os.path.join(matrix_route, 'rr.csr')

    # 2. 设置参数
    #fermi_energy = 13.063197611 # eV
    #occ_band = 4
    #omega_range = [0, 100] # eV
    #domega = 1 # eV
    kpt_grid = np.array([nkx, nky, nkz], dtype=int)

    """--------创建tight binding model-------"""
    m_tb = pyatb.init_tb(
        package = 'ABACUS',
        nspin = nspin,
        lattice_constant = lattice_constant,
        lattice_vector = lattice_vector,
        max_kpoint_num = 8000,
        isSparse = False,
        HR_route = HR_route,
        HR_unit = 'Ry',
        SR_route = SR_route,
        need_rR = True,
        rR_route = rR_route,
        rR_unit = 'Bohr',
        pR_route = pR_route,
        pR_unit= "Ry",
        need_pR= True
    )

    """----------------设置k点----------------"""
    k_start = np.array([0.0, 0.0, 0.0], dtype=float)
    k_vect1 = np.array([1.0, 0.0, 0.0], dtype=float)
    k_vect2 = np.array([0.0, 1.0, 0.0], dtype=float)
    k_vect3 = np.array([0.0, 0.0, 1.0], dtype=float)
    grid = kpt_grid
    kpt_grid_num = grid[0] * grid[1] * grid[2]
    kpt_generator = mp_generator(m_tb.max_kpoint_num, k_start, k_vect1, k_vect2,  k_vect3, grid)
    COMM.Barrier()
    basis_num = m_tb.basis_num
    for kpt in kpt_generator:
        ik_process = kpoints_in_different_process(SIZE, RANK, kpt)
        k_direct_coor_local = ik_process.k_direct_coor_local
        k_num = k_direct_coor_local.shape[0]

        if k_num:
            if(nspin==1 or nspin==4):
                eigenvalues, eigenvectors, velocity_matrix = m_tb.tb_solver.get_velocity_matrix(k_direct_coor_local)
                eigenvalues = [eigenvalues]
                eigenvectors = [eigenvectors]
                velocity_matrix = [velocity_matrix]
            if(nspin==2):
                eigenvalues_up, eigenvectors_up, velocity_matrix_up = m_tb.tb_solver_up.get_velocity_matrix(k_direct_coor_local)
                eigenvalues_dn, eigenvectors_dn, velocity_matrix_dn = m_tb.tb_solver_dn.get_velocity_matrix(k_direct_coor_local)
                eigenvalues = [eigenvalues_up, eigenvalues_dn]
                eigenvectors = [eigenvectors_up, eigenvectors_dn]
                velocity_matrix = [velocity_matrix_up, velocity_matrix_dn]
            #eigenvalues, pk_matrix = m_tb.tb_solver.get_pk_matrix(k_direct_coor_local)

    # 输出文件 only precision=16 for python float
    HA2EV = 27.211386245988
    if(use_soc):
        nspin = 1
    if RANK == 0:
        if (not os.path.exists("pyatb_librpa_df")):
            os.makedirs("pyatb_librpa_df")
        for ik in range(k_num):
            with open('pyatb_librpa_df/'+'KS_eigenvector_'+str(ik)+".dat", 'w') as f:
                f.write("%d"%(ik+1))
                f.write('\n')
                for ispin in range(nspin):
                    for ibasis in range(basis_num):
                        for iband in range(basis_num):
                            f.write("%30.16E%30.16E"%(eigenvectors[ispin][ik, ibasis, iband].real, eigenvectors[ispin][ik, ibasis, iband].imag))
                            f.write('\n')
        with open('pyatb_librpa_df/'+"k_path_info", 'w') as f:
            f.write("%8d%8d%8d%8d"%(basis_num,basis_num,nspin,k_num))
            f.write('\n')
            for ik in range(k_num):
                f.write("%30.16f%30.16f%30.16f"%(k_direct_coor_local[ik][0],k_direct_coor_local[ik][1],k_direct_coor_local[ik][2]))
                f.write('\n')

        with open('pyatb_librpa_df/'+"band_out", 'w') as f:
            f.write(str(k_num))
            f.write('\n')
            f.write(str(nspin))
            f.write('\n')
            f.write(str(basis_num))
            f.write('\n')
            f.write(str(basis_num))
            f.write('\n')
            f.write("%.6f"%(fermi_energy/HA2EV))
            f.write('\n')
            for ik in range(k_num):
                for ispin in range(nspin):
                    f.write("%3d%3d"%(ik+1,ispin+1))
                    f.write('\n')
                    for iband in range(basis_num):
                        if (iband < occ_band):
                            if(use_soc):
                                f.write("%3d%13.8f%30.16E%18.8f"%(iband+1,1.0,(eigenvalues[ispin][ik, iband]/HA2EV),eigenvalues[ispin][ik, iband]))
                            else:
                                if(nspin==2):
                                    f.write("%3d%13.8f%30.16E%18.8f"%(iband+1,1.0,(eigenvalues[ispin][ik, iband]/HA2EV),eigenvalues[ispin][ik, iband]))
                                else:
                                    f.write("%3d%13.8f%30.16E%18.8f"%(iband+1,2.0,(eigenvalues[ispin][ik, iband]/HA2EV),eigenvalues[ispin][ik, iband]))
                            f.write('\n')
                        else:
                            f.write("%3d%13.8f%30.16E%18.8f"%(iband+1,0.0,(eigenvalues[ispin][ik, iband]/HA2EV),eigenvalues[ispin][ik, iband]))
                            f.write('\n')
        with open('pyatb_librpa_df/'+"velocity_matrix", 'w') as f:
            f.write(str(k_num))
            f.write('\n')
            f.write(str(nspin))
            f.write('\n')
            f.write(str(basis_num))
            f.write('\n')
            f.write(str(basis_num))
            f.write('\n')
            for ispin in range(nspin):
                for ik in range(k_num):
                    for ialpha in range(3):
                        f.write("%5d%5d%5d"%(ialpha+1,ik+1,ispin+1))
                        f.write('\n')
                        for iband in range(basis_num):
                            for ibasis in range(basis_num):
                                f.write("%30.16E%30.16E"%(velocity_matrix[ispin][ik, ialpha, iband, ibasis].real, velocity_matrix[ispin][ik, ialpha, iband, ibasis].imag))
                                #if(iband==ibasis):
                                #    print(velocity_matrix[ik, ialpha, iband, ibasis])
                                f.write('\n')
        #with open('pyatb_librpa_df/'+"momentum_matrix", 'w') as f:
        #    f.write(str(k_num))
        #    f.write('\n')
        #    f.write(str(basis_num))
        #    f.write('\n')
        #    f.write(str(basis_num))
        #    f.write('\n')
        #    for ik in range(k_num):
        #        for ialpha in range(3):
        #            f.write("%5d%5d"%(ialpha+1,ik+1))
        #            f.write('\n')
        #            for iband in range(basis_num):
        #                for ibasis in range(basis_num):
        #                    f.write("%30.16E%30.16E"%(pk_matrix[ik, ialpha, iband, ibasis].real, pk_matrix[ik, ialpha, iband, ibasis].imag))
        #                    f.write('\n')
