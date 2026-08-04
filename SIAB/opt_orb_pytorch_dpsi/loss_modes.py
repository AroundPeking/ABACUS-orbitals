LOSS_MODES = frozenset(
    {
        "st_only",
        "st_constrained",
        "st_dpsi_joint",
        "pi_dpsi_joint",
        "pi_rpa_sensitive_joint",
    }
)

PROJECTED_PI_MODES = frozenset(
    {"pi_dpsi_joint", "pi_rpa_sensitive_joint"}
)
