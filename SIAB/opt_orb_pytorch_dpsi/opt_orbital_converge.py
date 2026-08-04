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
from radial_locality import RadialLocalityResult
from projected_pi_optimization import ProjectedPiOptimizationResult
from sternheimer_spillage import SternheimerLossResult

import math
import torch
import numpy as np


def _projected_pi_diagnostics(result, rank_tolerance):
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
	if getattr(result, "sensitivity_alpha", None) is not None:
		diagnostics.update(
			{
				"mode": "pi_rpa_sensitive_joint",
				"sensitivity_alpha": float(result.sensitivity_alpha),
				"family_power": int(result.family_power),
			}
		)
	return diagnostics


def _is_projected_pi_mode(mode):
	return mode in ("pi_dpsi_joint", "pi_rpa_sensitive_joint")


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
				if loss_config["low_frequency_guard_weight"] > 0.0:
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
		legacy_header = ("istep_big", "istep_small", "istep_all", "Spillage")

		for stage_index, info_opt in enumerate(self.info_optimize):
			new_loss_stage = stage_index in loss_configs
			pi_mode = (
				new_loss_stage
				and _is_projected_pi_mode(loss_configs[stage_index]["mode"])
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
						projected_pi_header
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
						if not isinstance(
							st_result, ProjectedPiOptimizationResult
						):
							raise TypeError(
								"projected-Pi evaluator must return "
								"ProjectedPiOptimizationResult"
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
					accepted = (
						constraints_ok
						and condition_ok
						and locality_condition_ok
						and low_frequency_ok
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
					violation_key = (
						max(
							dft_violation,
							dpsi_violation,
							condition_violation,
							locality_condition_violation,
							low_frequency_violation,
						),
						(
							dft_violation
							+ dpsi_violation
							+ condition_violation
							+ locality_condition_violation
							+ low_frequency_violation
						),
					)
					if best_violation is None or violation_key < best_violation["key"]:
						best_violation = {
							"key": violation_key,
							"dft": dft_violation,
							"dpsi": dpsi_violation,
							"condition": condition_violation,
							"locality_condition": locality_condition_violation,
							"low_frequency": low_frequency_violation,
							"max_st_condition": st_result.max_condition,
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
						components[
							"projected_pi" if pi_mode else "sternheimer"
						].item(),
					]
					if pi_mode:
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
					if pi_mode:
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
									loss_config[
										"projected_pi_rank_tolerance"
									],
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
				raise RuntimeError(
					"no accepted Sternheimer optimization point; smallest observed "
					"violation candidate: "
					f"dft={best_violation['dft']:.8g}, "
					f"dpsi={best_violation['dpsi']:.8g}, "
					f"low_frequency={best_violation['low_frequency']:.8g}, "
					f"condition={best_violation['condition']:.8g}; "
					f"locality_condition={best_violation['locality_condition']:.8g}; "
					f"max_st_condition={best_violation['max_st_condition']:.8g}, "
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
