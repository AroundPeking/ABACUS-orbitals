from opt_orbital_spillage import Opt_Orbital_Spillage
import IO.func_C
from opt_orbital import Opt_Orbital
import optimize
from freeze_orbitals import validate_freeze_orbitals, zero_frozen_gradients
from optimization_loss import (
	compose_loss,
	constraints_satisfied,
	low_frequency_guard_satisfied,
	normalize_loss_config,
	selection_component,
)
from loss_modes import PROJECTED_PI_MODES
from radial_locality import RadialLocalityResult
from projected_pi_optimization import ProjectedPiOptimizationResult
from sternheimer_spillage import SternheimerLossResult

import math
import torch
import numpy as np


def _finite_float(name, value):
	value = float(value)
	if not math.isfinite(value):
		raise ValueError(f"{name} must be finite")
	return value


def _finite_list(name, values):
	return [
		_finite_float(f"{name}[{index}]", value)
		for index, value in enumerate(values)
	]


def _finite_nonnegative_scalar_tensor(value, label):
	if (
		not isinstance(value, torch.Tensor)
		or value.ndim != 0
		or torch.is_complex(value)
		or not bool(torch.isfinite(value))
		or bool(value < 0.0)
	):
		raise ValueError(
			f"{label} is required and must be a finite nonnegative scalar tensor"
		)


def _finite_frequency_tensor(value, label, shape, nonnegative=False):
	if (
		not isinstance(value, torch.Tensor)
		or value.ndim != 1
		or value.shape != shape
		or torch.is_complex(value)
		or not bool(torch.all(torch.isfinite(value)))
		or (nonnegative and bool(torch.any(value < 0.0)))
	):
		raise ValueError(
			f"{label} is required and must be a finite frequency tensor"
		)


def _matches_configured_alpha(value, expected):
	if isinstance(value, bool):
		return False
	try:
		value = float(value)
	except (TypeError, ValueError):
		return False
	return math.isfinite(value) and value == expected


def _validate_projected_pi_result(result, loss_config, context):
	mode = loss_config["mode"]
	if not isinstance(result, ProjectedPiOptimizationResult):
		if mode == "pi_rpa_sensitive_joint":
			raise ValueError(
				f"{mode} {context} result must be ProjectedPiOptimizationResult"
			)
		raise TypeError(
			"projected-Pi evaluator must return ProjectedPiOptimizationResult"
		)
	if mode != "pi_rpa_sensitive_joint":
		return

	prefix = f"{mode} {context} result"
	expected_alpha = loss_config["projected_pi_sensitivity_alpha"]
	if not _matches_configured_alpha(result.sensitivity_alpha, expected_alpha):
		raise ValueError(
			f"{prefix} sensitivity_alpha must equal configured alpha "
			f"{expected_alpha}"
		)
	if (
		isinstance(result.family_power, bool)
		or not isinstance(result.family_power, int)
		or result.family_power != 4
	):
		raise ValueError(f"{prefix} family_power must be exactly 4")
	if not isinstance(result.family_results, dict) or set(
		result.family_results
	) != {"H", "H2"}:
		raise ValueError(
			f"{prefix} family_results must contain exactly H and H2"
		)

	_finite_nonnegative_scalar_tensor(result.loss, f"{prefix} total loss")
	_finite_float(f"{prefix} max_condition", result.max_condition)
	if (
		not isinstance(result.frequency_ha, torch.Tensor)
		or result.frequency_ha.ndim != 1
		or result.frequency_ha.numel() == 0
		or torch.is_complex(result.frequency_ha)
		or not bool(torch.all(torch.isfinite(result.frequency_ha)))
	):
		raise ValueError(
			f"{prefix} frequency_ha is required and must be a finite vector"
		)
	frequency_shape = result.frequency_ha.shape
	_finite_frequency_tensor(
		result.frequency_loss,
		f"{prefix} frequency_loss",
		frequency_shape,
		nonnegative=True,
	)

	reference_weight = None
	for name in ("H", "H2"):
		family = result.family_results[name]
		if not _matches_configured_alpha(
			getattr(family, "sensitivity_alpha", None), expected_alpha
		):
			raise ValueError(
				f"{prefix} {name} sensitivity_alpha must equal configured "
				f"alpha {expected_alpha}"
			)
		for field in ("loss", "base_loss", "sensitivity_loss"):
			_finite_nonnegative_scalar_tensor(
				getattr(family, field, None), f"{prefix} {name} {field}"
			)
		for field in (
			"frequency_loss",
			"frequency_base_loss",
			"frequency_sensitivity_loss",
		):
			_finite_frequency_tensor(
				getattr(family, field, None),
				f"{prefix} {name} {field}",
				frequency_shape,
				nonnegative=True,
			)
		for field in (
			"trace_log_difference",
			"minimum_reference_dielectric_eigenvalue",
			"minimum_candidate_dielectric_eigenvalue",
		):
			_finite_frequency_tensor(
				getattr(family, field, None),
				f"{prefix} {name} {field}",
				frequency_shape,
			)
		family_frequency = getattr(family, "frequency_ha", None)
		_finite_frequency_tensor(
			family_frequency,
			f"{prefix} {name} frequency_ha",
			frequency_shape,
		)
		if not torch.equal(family_frequency, result.frequency_ha):
			raise ValueError(
				f"{prefix} {name} frequency_ha must match aggregate frequency_ha"
			)
		weight = getattr(family, "frequency_weight", None)
		_finite_frequency_tensor(
			weight,
			f"{prefix} {name} frequency_weight",
			frequency_shape,
		)
		if bool(torch.any(weight <= 0.0)):
			raise ValueError(f"{prefix} {name} frequency_weight must be positive")
		if reference_weight is None:
			reference_weight = weight
		elif not torch.equal(weight, reference_weight):
			raise ValueError(
				f"{prefix} {name} frequency_weight must match H frequency_weight"
			)
		reference_rank = getattr(family, "reference_rank", None)
		if (
			isinstance(reference_rank, bool)
			or not isinstance(reference_rank, int)
			or reference_rank < 0
		):
			raise ValueError(f"{prefix} {name} reference_rank must be nonnegative")
		_finite_float(
			f"{prefix} {name} max_candidate_condition",
			getattr(family, "max_candidate_condition", math.nan),
		)


def _projected_pi_diagnostics(
	result, mode, rank_tolerance, loss_baseline=None
):
	if mode == "pi_rpa_sensitive_joint":
		if not isinstance(loss_baseline, dict):
			raise ValueError("RPA-sensitive diagnostics require loss baselines")
		for name in ("projected_pi_family", "projected_pi_h"):
			if name not in loss_baseline:
				raise ValueError(
					f"RPA-sensitive diagnostics require {name} baseline"
				)
		families = {}
		for name, family in result.family_results.items():
			families[name] = {
				"total_loss": _finite_float(f"{name} total loss", family.loss),
				"base_loss": _finite_float(f"{name} base loss", family.base_loss),
				"sensitivity_loss": _finite_float(
					f"{name} sensitivity loss", family.sensitivity_loss
				),
				"blend_loss": _finite_float(f"{name} blend loss", family.loss),
				"reference_rank": int(family.reference_rank),
				"max_candidate_condition": _finite_float(
					f"{name} maximum candidate condition",
					family.max_candidate_condition,
				),
				"frequency_total_loss": _finite_list(
					f"{name} frequency total loss", family.frequency_loss
				),
				"frequency_base_loss": _finite_list(
					f"{name} frequency base loss", family.frequency_base_loss
				),
				"frequency_sensitivity_loss": _finite_list(
					f"{name} frequency sensitivity loss",
					family.frequency_sensitivity_loss,
				),
				"trace_log_difference": _finite_list(
					f"{name} trace-log difference", family.trace_log_difference
				),
				"minimum_reference_dielectric_eigenvalue": _finite_list(
					f"{name} minimum reference dielectric eigenvalue",
					family.minimum_reference_dielectric_eigenvalue,
				),
				"minimum_candidate_dielectric_eigenvalue": _finite_list(
					f"{name} minimum candidate dielectric eigenvalue",
					family.minimum_candidate_dielectric_eigenvalue,
				),
			}
		h_family = result.family_results["H"]
		return {
			"mode": mode,
			"alpha": _finite_float("sensitivity alpha", result.sensitivity_alpha),
			"family_power": int(result.family_power),
			"initial_family_loss": _finite_float(
				"initial family loss", loss_baseline["projected_pi_family"]
			),
			"final_family_loss": _finite_float("final family loss", result.loss),
			"initial_h_loss": _finite_float(
				"initial H loss", loss_baseline["projected_pi_h"]
			),
			"final_h_loss": _finite_float("final H loss", h_family.loss),
			"frequency_ha": _finite_list("frequency", result.frequency_ha),
			"frequency_weight": _finite_list(
				"frequency weight", h_family.frequency_weight
			),
			"frequency_total_loss": _finite_list(
				"frequency total loss", result.frequency_loss
			),
			"max_overlap_condition": _finite_float(
				"maximum overlap condition", result.max_condition
			),
			"rank_tolerance": _finite_float("rank tolerance", rank_tolerance),
			"family_names": list(result.family_results),
			"families": families,
		}

	families = {}
	for name, family in result.family_results.items():
		record = {}
		if hasattr(family, "loss"):
			record["loss"] = float(family.loss)
		if hasattr(family, "reference_rank"):
			record["reference_rank"] = int(family.reference_rank)
		if hasattr(family, "max_candidate_condition"):
			record["max_candidate_condition"] = float(
				family.max_candidate_condition
			)
		if hasattr(family, "frequency_loss"):
			record["frequency_loss"] = [
				float(value) for value in family.frequency_loss
			]
		families[name] = record
	diagnostics = {
		"frequency_ha": [float(value) for value in result.frequency_ha],
		"frequency_loss": [float(value) for value in result.frequency_loss],
		"lowest_frequency_ha": float(result.lowest_frequency_ha),
		"lowest_frequency_loss": float(result.lowest_frequency_loss),
		"max_condition": float(result.max_condition),
		"rank_tolerance": float(rank_tolerance),
		"family_names": list(result.family_results),
		"families": families,
	}
	return diagnostics


def _is_projected_pi_mode(mode):
	return mode in PROJECTED_PI_MODES


class Opt_Orbital_Converge:
	def set_info(self, file_list, info_optimize, info_stru, info_C_init, info_V):
		self.file_list = file_list
		self.info_optimize = info_optimize
		self.info_stru = info_stru
		self.info_C_init = info_C_init
		self.info_V = info_V

	def set_info_element(self, info_element):
		self.info_element = info_element

	def set_QSVI(self, QI, SI, VI_origin):
		self.QI = QI
		self.SI = SI
		self.VI_origin = VI_origin

	def set_QSVI_linear(self, QI_linear, SI_linear, VI_linear):
		self.QI_linear = QI_linear
		self.SI_linear = SI_linear
		self.VI_linear = VI_linear

	def set_C_read_index(self, C_read_index):
		self.C_read_index = C_read_index

	def set_E(self, E):
		self.E = E

	def set_sternheimer_spillage(self, sternheimer_spillage):
		self.sternheimer_spillage = sternheimer_spillage

	def set_projected_pi_objective(self, projected_pi_objective):
		self.projected_pi_objective = projected_pi_objective

	def set_radial_locality(self, radial_locality):
		self.radial_locality = radial_locality

	def _make_spillage(self, info_opt):
		if not hasattr(self, "QI"):
			raise ValueError("legacy DFT spillage requires origin data")
		spillage = Opt_Orbital_Spillage(
			self.info_stru,
			self.info_element,
			self.info_V,
			info_opt["norm"],
			self.file_list,
		)
		spillage.set_QSVI(self.QI, self.SI, self.VI_origin)
		if "linear" in self.file_list.keys():
			spillage.set_QSVI_linear(self.QI_linear, self.SI_linear, self.VI_linear)
		return spillage

	def cal_converge(self, C, files):
		data_transmit = dict()
		explicit_freeze = "freeze_orbitals" in self.info_C_init
		if explicit_freeze:
			freeze_indices = validate_freeze_orbitals(
				self.info_C_init["freeze_orbitals"], C
			)
		else:
			freeze_indices = frozenset()
		C_initial = IO.func_C.copy_C(C, self.info_element)

		def restore_frozen_columns():
			if not explicit_freeze:
				return
			with torch.no_grad():
				for it, il, iu in freeze_indices:
					C[it][il][:, iu].copy_(C_initial[it][il][:, iu])

		loss_configs = dict()
		loss_baselines = dict()
		projected_pi_baseline_cache = dict()
		new_stage_indices = []
		for stage_index, info_opt in enumerate(self.info_optimize):
			if "loss" not in info_opt:
				continue
			new_stage_indices.append(stage_index)
			loss_configs[stage_index] = normalize_loss_config(info_opt["loss"])
			if info_opt["cal_T"]:
				raise ValueError(
					"cal_T=True is not supported for Sternheimer loss stages"
				)

		if new_stage_indices:
			for stage_index in new_stage_indices:
				loss_config = loss_configs[stage_index]
				if _is_projected_pi_mode(loss_config["mode"]):
					evaluator = getattr(self, "projected_pi_objective", None)
					if not callable(getattr(evaluator, "evaluate", None)):
						raise ValueError(
							f"{loss_config['mode']} requires a projected-Pi evaluator; "
							"call set_projected_pi_objective first"
						)
				else:
					evaluator = getattr(self, "sternheimer_spillage", None)
					if not callable(getattr(evaluator, "evaluate", None)):
						raise ValueError(
							"Sternheimer loss stage requires a Sternheimer "
							"evaluator; call set_sternheimer_spillage first"
						)
				if loss_config["radial_tail_radius"] > 0.0:
					locality = getattr(self, "radial_locality", None)
					if not callable(getattr(locality, "evaluate", None)):
						raise ValueError(
							"positive radial_tail_radius requires a radial locality evaluator; "
							"call set_radial_locality first"
						)
				if hasattr(self, "QI"):
					baseline_spillage = self._make_spillage(
						self.info_optimize[stage_index]
					)
					baseline_components = baseline_spillage.cal_components(C_initial)
				else:
					if loss_config["mode"] != "st_only":
						raise ValueError(
							"st_constrained, st_dpsi_joint, pi_dpsi_joint, and "
							"pi_rpa_sensitive_joint "
							"require legacy "
							"DFT and dpsi data"
						)
					zero = next(iter(C_initial.values()))[0].sum() * 0.0
					baseline_components = {
						"dft_origin": zero,
						"dft_dpsi": zero,
					}
				loss_baselines[stage_index] = {
					"dft_origin": baseline_components["dft_origin"].detach().clone(),
					"dft_dpsi": baseline_components["dft_dpsi"].detach().clone(),
				}
				if loss_config["mode"] == "pi_rpa_sensitive_joint":
					baseline_key = (
						id(C_initial),
						id(evaluator),
						tuple(
							(name, loss_config[name])
							for name in sorted(loss_config)
						),
					)
					baseline_record = projected_pi_baseline_cache.get(
						baseline_key
					)
					if baseline_record is None:
						with torch.no_grad():
							baseline_result = evaluator.evaluate(C_initial)
							_validate_projected_pi_result(
								baseline_result, loss_config, "baseline"
							)
							family_baseline = baseline_result.loss.detach().clone()
							h_baseline = baseline_result.family_results[
								"H"
							].loss.detach().clone()
							baseline_record = {
								"projected_pi_family": family_baseline,
								"projected_pi_h": h_baseline,
								"initial_diagnostics": {
									"family_loss": family_baseline.item(),
									"h_loss": h_baseline.item(),
								},
							}
						del baseline_result
						zero_baselines = []
						if h_baseline.item() == 0.0:
							zero_baselines.append("H")
						if family_baseline.item() == 0.0:
							zero_baselines.append("family")
						if zero_baselines:
							raise RuntimeError(
								"pi_rpa_sensitive_joint strict improvement "
								"impossible: baseline "
								+ " and ".join(zero_baselines)
								+ " loss is zero"
							)
						projected_pi_baseline_cache[baseline_key] = (
							baseline_record
						)
					loss_baselines[stage_index]["projected_pi_family"] = (
						baseline_record["projected_pi_family"].clone()
					)
					loss_baselines[stage_index]["projected_pi_h"] = (
						baseline_record["projected_pi_h"].clone()
					)
				elif loss_config["low_frequency_guard_weight"] > 0.0:
					baseline_st = evaluator.evaluate(C_initial)
					loss_baselines[stage_index][
						"sternheimer_lowest_frequency"
					] = baseline_st.lowest_frequency_loss.detach().clone()

		best_accepted = None
		best_violation = None
		detail_schema = None
		new_header = (
			"istep_big", "istep_small", "istep_all", "dft_origin",
			"dft_dpsi", "sternheimer", "regularization_dpsi",
			"constraint_dft", "constraint_dpsi", "total", "radial_tail",
			"regularization_locality", "max_st_condition",
			"max_locality_condition", "accepted",
		)
		guarded_header = (
			"istep_big", "istep_small", "istep_all", "dft_origin",
			"dft_dpsi", "sternheimer", "sternheimer_lowest_frequency",
			"regularization_low_frequency", "regularization_dpsi",
			"constraint_dft", "constraint_dpsi", "total", "radial_tail",
			"regularization_locality", "max_st_condition",
			"max_locality_condition", "accepted",
		)
		projected_pi_header = (
			"istep_big", "istep_small", "istep_all", "dft_origin",
			"dft_dpsi", "projected_pi",
			"projected_pi_lowest_frequency", "regularization_dpsi",
			"constraint_dft", "constraint_dpsi", "total",
			"max_projected_pi_condition", "accepted",
		)
		rpa_sensitive_header = (
			"istep_big", "istep_small", "istep_all", "dft_origin",
			"dft_dpsi", "projected_pi_family", "projected_pi_h_total",
			"projected_pi_h_base", "projected_pi_h_sensitivity",
			"projected_pi_h_blend", "projected_pi_h2_total",
			"projected_pi_h2_base", "projected_pi_h2_sensitivity",
			"projected_pi_h2_blend", "regularization_dpsi",
			"constraint_dft", "constraint_dpsi", "total",
			"family_improved", "atom_improved",
			"max_projected_pi_condition", "accepted",
		)
		legacy_header = ("istep_big", "istep_small", "istep_all", "Spillage")

		for stage_index, info_opt in enumerate(self.info_optimize):
			new_loss_stage = stage_index in loss_configs
			pi_mode = (
				new_loss_stage
				and _is_projected_pi_mode(loss_configs[stage_index]["mode"])
			)
			rpa_sensitive_mode = (
				new_loss_stage
				and loss_configs[stage_index]["mode"]
				== "pi_rpa_sensitive_joint"
			)
			guard_active = (
				new_loss_stage
				and loss_configs[stage_index]["low_frequency_guard_weight"] > 0.0
			)
			print( 'See "Spillage.dat" for detail status:', file=files[0], flush=True )
			print( "istep", "Spillage", sep="\t", file=files[0], flush=True )
			if new_loss_stage:
				print(
					*(
						rpa_sensitive_header
						if rpa_sensitive_mode
						else projected_pi_header
						if pi_mode
						else guarded_header if guard_active else new_header
					),
					sep="\t",
					file=files[1],
				)
				detail_schema = "new"
			elif detail_schema == "new":
				print(*legacy_header, sep="\t", file=files[1])
				detail_schema = "legacy"

			opt = optimize.get_optim(info_opt, sum(C.values(), []))
			spillage = self._make_spillage(info_opt) if hasattr(self, "QI") else None

			if new_loss_stage:
				loss_config = loss_configs[stage_index]
				loss_baseline = loss_baselines[stage_index]

				def calculate_components():
					if pi_mode:
						st_result = self.projected_pi_objective.evaluate(C)
						_validate_projected_pi_result(
							st_result, loss_config, "candidate"
						)
					else:
						st_result = self.sternheimer_spillage.evaluate(C)
						if not isinstance(st_result, SternheimerLossResult):
							raise TypeError(
								"Sternheimer evaluator must return "
								"SternheimerLossResult"
							)
					if not math.isfinite(st_result.max_condition):
						label = (
							"max_projected_pi_condition"
							if pi_mode
							else "max_st_condition"
						)
						raise ValueError(f"{label} must be finite")
					if spillage is None:
						zero = torch.zeros_like(st_result.loss)
						legacy_components = {
							"dft_origin": zero,
							"dft_dpsi": zero,
						}
					else:
						legacy_components = spillage.cal_components(C)
					if loss_config["radial_tail_radius"] > 0.0:
						locality_result = self.radial_locality.evaluate(C)
						if not isinstance(locality_result, RadialLocalityResult):
							raise TypeError(
								"radial locality evaluator must return RadialLocalityResult"
							)
						if not math.isfinite(locality_result.max_condition):
							raise ValueError(
								"max_locality_condition must be finite"
							)
					else:
						locality_result = RadialLocalityResult(
							loss=torch.zeros_like(st_result.loss),
							max_condition=1.0,
							by_channel={},
						)
					components = compose_loss(
						loss_config["mode"],
						st_result.loss,
						legacy_components["dft_origin"],
						legacy_components["dft_dpsi"],
						loss_baseline,
						loss_config,
						radial_tail=locality_result.loss,
						st_low_frequency=(
							st_result.lowest_frequency_loss
							if guard_active
							else None
						),
					)
					return (
						legacy_components,
						st_result,
						locality_result,
						components,
					)

				def evaluate_candidate(istep_big, closure_count):
					nonlocal best_accepted, best_violation
					with torch.no_grad():
						(
							legacy_components,
							st_result,
							locality_result,
							components,
						) = calculate_components()
						constraints_ok = (
							loss_config["mode"] == "st_only"
							or constraints_satisfied(
								legacy_components["dft_origin"],
								legacy_components["dft_dpsi"],
								loss_baseline,
								loss_config,
							)
						)
						condition_ok = (
							st_result.max_condition <= loss_config["condition_limit"]
						)
						locality_condition_ok = (
							locality_result.max_condition
							<= loss_config["radial_tail_condition_limit"]
						)
						low_frequency_ok = low_frequency_guard_satisfied(
							(
								st_result.lowest_frequency_loss
								if guard_active
								else st_result.loss
							),
							loss_baseline,
							loss_config,
						)
						if rpa_sensitive_mode:
							family_current = st_result.loss.item()
							family_baseline = loss_baseline[
								"projected_pi_family"
							].item()
							atom_current = st_result.family_results["H"].loss.item()
							atom_baseline = loss_baseline["projected_pi_h"].item()
							family_improved = bool(
								family_current < family_baseline
							)
							atom_improved = bool(
								atom_current < atom_baseline
							)
							family_signed_delta = (
								family_current - family_baseline
							) / max(family_baseline, loss_config["epsilon"])
							atom_signed_delta = (
								atom_current - atom_baseline
							) / max(atom_baseline, loss_config["epsilon"])
							response_failed_gates = int(not family_improved) + int(
								not atom_improved
							)
							family_gate_penalty = (
								0.0
								if family_improved
								else 1.0 + max(0.0, family_signed_delta)
							)
							atom_gate_penalty = (
								0.0
								if atom_improved
								else 1.0 + max(0.0, atom_signed_delta)
							)
						else:
							family_improved = True
							atom_improved = True
							family_current = None
							family_baseline = None
							atom_current = None
							atom_baseline = None
							family_signed_delta = 0.0
							atom_signed_delta = 0.0
							response_failed_gates = 0
							family_gate_penalty = 0.0
							atom_gate_penalty = 0.0
						accepted = (
							constraints_ok
							and condition_ok
							and locality_condition_ok
							and low_frequency_ok
							and family_improved
							and atom_improved
						)

						baseline_dft = max(
							loss_baseline["dft_origin"].item(), loss_config["epsilon"]
						)
						baseline_dpsi = max(
							loss_baseline["dft_dpsi"].item(), loss_config["epsilon"]
						)
						dft_violation = max(
							0.0,
							legacy_components["dft_origin"].item() / baseline_dft
							- 1.0 - loss_config["tau_dft"],
						)
						dpsi_violation = max(
							0.0,
							legacy_components["dft_dpsi"].item() / baseline_dpsi
							- 1.0 - loss_config["tau_dpsi"],
						)
						condition_violation = max(
							0.0,
							st_result.max_condition / loss_config["condition_limit"] - 1.0,
						)
						locality_condition_violation = max(
							0.0,
							locality_result.max_condition
							/ loss_config["radial_tail_condition_limit"]
							- 1.0,
						)
						if guard_active:
							baseline_low_frequency = loss_baseline[
								"sternheimer_lowest_frequency"
							].item()
							low_frequency_violation = max(
								0.0,
								st_result.lowest_frequency_loss.item()
								/ baseline_low_frequency
								- 1.0
								- loss_config["low_frequency_guard_tolerance"],
							)
						else:
							low_frequency_violation = 0.0
						base_violations = (
							dft_violation,
							dpsi_violation,
							condition_violation,
							locality_condition_violation,
							low_frequency_violation,
						)
						if rpa_sensitive_mode:
							all_penalties = base_violations + (
								family_gate_penalty,
								atom_gate_penalty,
							)
							failed_gate_count = sum(
								violation > 0.0 for violation in base_violations
							) + response_failed_gates
							violation_key = (
								failed_gate_count,
								max(all_penalties),
								sum(all_penalties),
								family_signed_delta + atom_signed_delta,
								family_signed_delta,
								atom_signed_delta,
							)
						else:
							violation_key = (
								max(base_violations),
								sum(base_violations),
							)
						if best_violation is None or violation_key < best_violation["key"]:
							best_violation = {
								"key": violation_key,
								"mode": loss_config["mode"],
								"dft": dft_violation,
								"dpsi": dpsi_violation,
								"atom": atom_gate_penalty,
								"family": family_gate_penalty,
								"response_failed_gates": response_failed_gates,
								"family_improved": family_improved,
								"atom_improved": atom_improved,
								"family_current": family_current,
								"family_baseline": family_baseline,
								"family_signed_delta": family_signed_delta,
								"atom_current": atom_current,
								"atom_baseline": atom_baseline,
								"atom_signed_delta": atom_signed_delta,
								"condition": condition_violation,
								"locality_condition": locality_condition_violation,
								"low_frequency": low_frequency_violation,
								"condition_label": (
									"max_projected_pi_condition"
									if pi_mode
									else "max_st_condition"
								),
								"max_condition": st_result.max_condition,
								"condition_limit": loss_config["condition_limit"],
								"max_locality_condition": locality_result.max_condition,
								"locality_condition_limit": (
									loss_config["radial_tail_condition_limit"]
								),
							}

						row = [
							istep_big,
							closure_count,
							data_transmit["istep_all"],
							components["dft_origin"].item(),
							components["dft_dpsi"].item(),
						]
						if rpa_sensitive_mode:
							h = st_result.family_results["H"]
							h2 = st_result.family_results["H2"]
							row.extend(
								[
									components["projected_pi"].item(),
									h.loss.item(),
									h.base_loss.item(),
									h.sensitivity_loss.item(),
									h.loss.item(),
									h2.loss.item(),
									h2.base_loss.item(),
									h2.sensitivity_loss.item(),
									h2.loss.item(),
								]
							)
						else:
							row.append(
								components[
									"projected_pi" if pi_mode else "sternheimer"
								].item()
							)
						if pi_mode and not rpa_sensitive_mode:
							row.append(st_result.lowest_frequency_loss.item())
						elif guard_active:
							row.extend(
								[
									components[
										"sternheimer_lowest_frequency"
									].item(),
									components[
										"regularization_low_frequency"
									].item(),
								]
							)
						row.extend(
							[
								components["regularization_dpsi"].item(),
								components["constraint_dft"].item(),
								components["constraint_dpsi"].item(),
								components["total"].item(),
							]
						)
						if rpa_sensitive_mode:
							row.extend(
								[
									str(family_improved).lower(),
									str(atom_improved).lower(),
									st_result.max_condition,
									str(accepted).lower(),
								]
							)
						elif pi_mode:
							row.extend(
								[st_result.max_condition, str(accepted).lower()]
							)
						else:
							row.extend(
								[
									components["radial_tail"].item(),
									components["regularization_locality"].item(),
									st_result.max_condition,
									locality_result.max_condition,
									str(accepted).lower(),
								]
							)
						print(*row, sep="\t", file=files[1])
						data_transmit["istep_all"] += 1
						data_transmit.update(
							{
								"Loss": components["total"].item(),
								"Spillage": components["total"].item(),
							}
						)

						selection_name = selection_component(loss_config["mode"])
					if accepted and (
						best_accepted is None
						or components[selection_name].item()
						< best_accepted["loss_components"][selection_name]
					):
						best_accepted = {
							"C": IO.func_C.copy_C(C, self.info_element),
							"loss_components": {
								name: value.item() for name, value in components.items()
							},
							"loss_mode": loss_config["mode"],
							"loss_baseline": {
								name: value.item() for name, value in loss_baseline.items()
							},
						}
						if pi_mode:
							best_accepted["max_projected_pi_condition"] = (
								st_result.max_condition
							)
							best_accepted["projected_pi_diagnostics"] = (
								_projected_pi_diagnostics(
									st_result,
									loss_config["mode"],
									loss_config[
										"projected_pi_rank_tolerance"
									],
									loss_baseline if rpa_sensitive_mode else None,
								)
							)
						else:
							best_accepted["max_st_condition"] = (
								st_result.max_condition
							)
							best_accepted["max_locality_condition"] = (
								locality_result.max_condition
							)
						if guard_active:
							best_accepted["low_frequency_diagnostics"] = {
								"lowest_st_frequency_ha": (
									st_result.lowest_frequency_ha.item()
								),
								"initial_lowest_st_loss": loss_baseline[
									"sternheimer_lowest_frequency"
								].item(),
								"final_lowest_st_loss": (
									st_result.lowest_frequency_loss.item()
								),
								"low_frequency_guard_tolerance": loss_config[
									"low_frequency_guard_tolerance"
								],
								"low_frequency_guard_weight": loss_config[
									"low_frequency_guard_weight"
								],
							}
						data_transmit["flag_finish"] = 0
					else:
						data_transmit["flag_finish"] += 1

				def closure():
					opt.zero_grad()
					_, _, _, components = calculate_components()
					Loss = components["total"]
					Loss.backward()
					if explicit_freeze:
						zero_frozen_gradients(C, freeze_indices)
					elif hasattr(self, "C_read_index"):
						for it, il, iu in self.C_read_index:
							C[it][il].grad[:, iu] = 0
					return Loss

				data_transmit["istep_all"] = 0
				data_transmit["flag_finish"] = 0
				evaluate_candidate(-1, 0)
				for data_transmit["istep_big"] in range(info_opt["max_steps"]):
					if (
						info_opt["optimizer"] != "LBFGS"
						and data_transmit["flag_finish"] > 50
					):
						break
					closure_count = 0

					def counted_closure():
						nonlocal closure_count
						closure_count += 1
						return closure()

					opt.step(counted_closure)
					restore_frozen_columns()
					evaluate_candidate(data_transmit["istep_big"], closure_count)
					if (
						info_opt["optimizer"] == "LBFGS"
						or data_transmit["istep_big"] % 100 == 0
					):
						print(
							data_transmit["istep_big"],
							data_transmit["Spillage"],
							sep="\t",
							file=files[0],
							flush=True,
						)
					if info_opt["optimizer"] == "LBFGS" and closure_count == 1:
						break
				continue

			def closure():
				Spillage = spillage.cal_Spillage(C)
				if info_opt["cal_T"]:
					T = Opt_Orbital.cal_T(C, self.E)
					if not "TSrate" in vars():	TSrate = torch.abs(0.002*Spillage/T).data[0]
					Loss = Spillage + TSrate*T
				else:
					Loss = Spillage
				if info_opt["cal_T"]:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item(), T.item(), Loss.item()]
				else:
					print_content = [data_transmit["istep_big"], data_transmit["istep_small"], data_transmit["istep_all"], Spillage.item()]
				print(*print_content, sep="\t", file=files[1])
				data_transmit["istep_small"] += 1
				data_transmit["istep_all"] += 1
				data_transmit.update({"Loss":Loss.item(), "Spillage":Spillage.item()})
				if info_opt["optimizer"] != "LBFGS":
					if Loss < data_transmit["loss_saved"]:
						data_transmit["loss_saved"] = Loss
						data_transmit["flag_finish"] = 0
						data_transmit["C"] = IO.func_C.copy_C(C,self.info_element)
					else:
						data_transmit["flag_finish"] += 1
				else:
					data_transmit["C"] = IO.func_C.copy_C(C,self.info_element)
				opt.zero_grad()
				Loss.backward()
				if explicit_freeze:
					zero_frozen_gradients(C, freeze_indices)
				elif hasattr(self, "C_read_index"):
					for it,il,iu in self.C_read_index:
						C[it][il].grad[:,iu] = 0
				return Loss

			data_transmit["istep_all"] = 0
			if info_opt["optimizer"] != "LBFGS":
				data_transmit["loss_saved"] = np.inf
				data_transmit["flag_finish"] = 0
			for data_transmit["istep_big"] in range(info_opt["max_steps"]):
				data_transmit["istep_small"] = 0
				if info_opt["optimizer"] != "LBFGS":
					if data_transmit["flag_finish"] > 50:
						break
				opt.step(closure)
				restore_frozen_columns()
				if (info_opt["optimizer"]=="LBFGS") or (data_transmit["istep_big"]%100==0):
					print(data_transmit["istep_big"], data_transmit["Spillage"], sep="\t", file=files[0], flush=True)
				if info_opt["optimizer"] == "LBFGS":
					if data_transmit["istep_small"]==1:
						break

		if new_stage_indices:
			if best_accepted is None:
				if best_violation["mode"] == "pi_rpa_sensitive_joint":
					response_diagnostics = (
						f"response_failed_gates="
						f"{best_violation['response_failed_gates']}, "
						f"family_improved="
						f"{str(best_violation['family_improved']).lower()}, "
						f"atom_improved="
						f"{str(best_violation['atom_improved']).lower()}; "
						f"atom_current={best_violation['atom_current']:.8g}, "
						f"atom_baseline={best_violation['atom_baseline']:.8g}, "
						f"atom_signed_normalized_delta="
						f"{best_violation['atom_signed_delta']:.8g}; "
						f"family_current={best_violation['family_current']:.8g}, "
						f"family_baseline={best_violation['family_baseline']:.8g}, "
						f"family_signed_normalized_delta="
						f"{best_violation['family_signed_delta']:.8g}; "
					)
				else:
					response_diagnostics = (
						f"atom={best_violation['atom']:.8g}, "
						f"family={best_violation['family']:.8g}, "
					)
				raise RuntimeError(
					"no accepted Sternheimer optimization point; smallest observed "
					"violation candidate: "
					f"dft={best_violation['dft']:.8g}, "
					f"dpsi={best_violation['dpsi']:.8g}, "
					+ response_diagnostics
					+ f"low_frequency={best_violation['low_frequency']:.8g}, "
					f"condition={best_violation['condition']:.8g}; "
					f"locality_condition={best_violation['locality_condition']:.8g}; "
					f"{best_violation['condition_label']}="
					f"{best_violation['max_condition']:.8g}, "
					f"condition_limit={best_violation['condition_limit']:.8g}, "
					f"max_locality_condition="
					f"{best_violation['max_locality_condition']:.8g}, "
					f"locality_condition_limit="
					f"{best_violation['locality_condition_limit']:.8g}"
				)
			data_transmit.update(best_accepted)
			data_transmit["Loss"] = best_accepted["loss_components"]["total"]
			data_transmit["Spillage"] = best_accepted["loss_components"]["total"]
		return data_transmit
