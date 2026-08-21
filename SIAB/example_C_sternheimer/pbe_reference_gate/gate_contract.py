from __future__ import annotations


VALID_MODES = {"fixed", "field", "free"}


def render_input(
    *, mode: str, field_dir: int | None = None, restart: bool = False
) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    if mode in {"field", "free"} and field_dir not in {0, 1, 2}:
        raise ValueError("field_dir must be 0, 1, or 2")
    if mode == "fixed" and field_dir is not None:
        raise ValueError("fixed mode does not accept field_dir")

    values = [
        ("INPUT_PARAMETERS", None),
        ("suffix", "C_PBE_REFERENCE_GATE"),
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
        ("kpar", "1"),
        ("pseudo_dir", "./"),
        ("orbital_dir", "./"),
        ("scf_thr", "1e-10"),
        ("scf_nmax", "300"),
        ("mixing_type", "broyden"),
        ("mixing_beta", "0.3"),
        ("mixing_beta_mag", "0.3"),
        ("smearing_method", "fixed"),
        ("ocp", "1" if mode == "fixed" else "0"),
    ]
    if mode == "fixed":
        values.append(("ocp_set", "3*1 19*0 1*1 21*0"))
    if mode == "field":
        values.extend(
            [
                ("efield_flag", "1"),
                ("dip_cor_flag", "0"),
                ("efield_dir", str(field_dir)),
                ("efield_pos_max", "0.8"),
                ("efield_pos_dec", "0.1"),
                ("efield_amp", "1e-4"),
            ]
        )
    else:
        values.extend([("efield_flag", "0"), ("efield_amp", "0")])
    values.extend(
        [("out_chg", "1"), ("out_wfc_lcao", "2"), ("out_mul", "1")]
    )
    if restart:
        values.extend([("init_wfc", "file"), ("init_chg", "file")])

    return "\n".join(
        key if value is None else f"{key} {value}" for key, value in values
    ) + "\n"
