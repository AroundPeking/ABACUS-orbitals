from __future__ import annotations


SIAB_PROTOCOL = (
    ("suffix", "C_SIAB_REFERENCE"),
    ("calculation", "scf"),
    ("ntype", "1"),
    ("nelec", "4"),
    ("nspin", "2"),
    ("nupdown", "2"),
    ("nbands", "22"),
    ("basis_type", "lcao"),
    ("ecutwfc", "30"),
    ("lcao_ecut", "100"),
    ("nx", "135"),
    ("ny", "135"),
    ("nz", "135"),
    ("ks_solver", "genelpa"),
    ("dft_functional", "pbe"),
    ("symmetry", "0"),
    ("gamma_only", "1"),
    ("kpar", "1"),
    ("pseudo_dir", "./"),
    ("orbital_dir", "./"),
    ("scf_thr", "1e-10"),
    ("scf_nmax", "300"),
    ("mixing_type", "broyden"),
    ("mixing_beta", "0.3"),
    ("mixing_beta_mag", "0.3"),
    ("smearing_method", "fixed"),
    ("ocp", "1"),
    ("ocp_set", "3*1 19*0 1*1 21*0"),
    ("efield_flag", "0"),
    ("efield_amp", "0"),
    ("init_wfc", "file"),
    ("init_chg", "file"),
    ("out_chg", "1"),
    ("out_wfc_lcao", "1"),
    ("out_app_flag", "1"),
    ("out_mul", "1"),
    ("bessel_nao_ecut", "100"),
    ("bessel_nao_rcut", "10"),
    ("bessel_nao_smooth", "1"),
    ("bessel_nao_sigma", "0.1"),
    ("bessel_nao_tolerence", "1e-12"),
    ("out_librpa_reader_version", "1"),
    ("out_sternheimer_librpa", "0"),
    ("out_sternheimer_siab", "1"),
    ("sternheimer_siab_coulomb_threshold", "1e-10"),
    ("sternheimer_siab_lmax", "4"),
    ("sternheimer_nfreq", "16"),
    ("sternheimer_frequency_grid_file", "fixed_frequency_grid_nfreq16.dat"),
    ("sternheimer_frequency_mpi", "1"),
    ("sternheimer_channel_mpi", "1"),
    ("sternheimer_delta", "1"),
    ("sternheimer_fd_order", "8"),
    ("sternheimer_delta_max_states", "0"),
    ("sternheimer_delta_norm_tol", "1e-10"),
    # The explicit ABFS is already the PCA=1e-4 product.  A value above one
    # disables a second, inconsistent PCA truncation inside ABACUS.
    ("exx_pca_threshold", "10"),
    ("exx_singularity_correction", "massidda"),
    ("exx_ccp_rmesh_times", "1"),
    ("rpa_ccp_rmesh_times", "1"),
)


def render_siab_input() -> str:
    return "INPUT_PARAMETERS\n" + "\n".join(
        f"{key} {value}" for key, value in SIAB_PROTOCOL
    ) + "\n"


def render_siab_stru(source: str, abfs_name: str) -> str:
    if "ABFS_ORBITAL" in source:
        raise ValueError("source STRU already contains ABFS_ORBITAL")
    if not abfs_name or any(character.isspace() for character in abfs_name):
        raise ValueError("ABFS filename must be one nonempty token")
    return source.rstrip() + f"\n\nABFS_ORBITAL\n{abfs_name}\n"
