from opt_orbital_spillage import Opt_Orbital_Spillage
import IO.func_C
from opt_orbital import Opt_Orbital
import optimize
from freeze_orbitals import validate_freeze_orbitals, zero_frozen_gradients
from optimization_loss import (
	compose_loss,
	constraints_satisfied,
	normalize_loss_config,
)
from sternheimer_spillage import SternheimerLossResult

import math
import torch
import numpy as np


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

	def _make_spillage(self, info_opt):
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
			evaluator = getattr(self, "sternheimer_spillage", None)
			if not callable(getattr(evaluator, "evaluate", None)):
				raise ValueError(
					"Sternheimer loss stage requires a Sternheimer evaluator; "
					"call set_sternheimer_spillage first"
				)
			for stage_index in new_stage_indices:
				baseline_spillage = self._make_spillage(
					self.info_optimize[stage_index]
				)
				baseline_components = baseline_spillage.cal_components(C_initial)
				loss_baselines[stage_index] = {
					"dft_origin": baseline_components["dft_origin"].detach().clone(),
					"dft_dpsi": baseline_components["dft_dpsi"].detach().clone(),
				}

		best_accepted = None
		best_violation = None
		detail_schema = None
		new_header = (
			"istep_big", "istep_small", "istep_all", "dft_origin",
			"dft_dpsi", "sternheimer", "constraint_dft", "constraint_dpsi",
			"total", "max_st_condition", "accepted",
		)
		legacy_header = ("istep_big", "istep_small", "istep_all", "Spillage")

		for stage_index, info_opt in enumerate(self.info_optimize):
			new_loss_stage = stage_index in loss_configs
			print( 'See "Spillage.dat" for detail status:', file=files[0], flush=True )
			print( "istep", "Spillage", sep="\t", file=files[0], flush=True )
			if new_loss_stage:
				print(*new_header, sep="\t", file=files[1])
				detail_schema = "new"
			elif detail_schema == "new":
				print(*legacy_header, sep="\t", file=files[1])
				detail_schema = "legacy"

			opt = optimize.get_optim(info_opt, sum(C.values(), []))
			spillage = self._make_spillage(info_opt)

			if new_loss_stage:
				loss_config = loss_configs[stage_index]
				loss_baseline = loss_baselines[stage_index]

				def calculate_components():
					legacy_components = spillage.cal_components(C)
					st_result = self.sternheimer_spillage.evaluate(C)
					if not isinstance(st_result, SternheimerLossResult):
						raise TypeError(
							"Sternheimer evaluator must return SternheimerLossResult"
						)
					if not math.isfinite(st_result.max_condition):
						raise ValueError("max_st_condition must be finite")
					components = compose_loss(
						loss_config["mode"],
						st_result.loss,
						legacy_components["dft_origin"],
						legacy_components["dft_dpsi"],
						loss_baseline,
						loss_config,
					)
					return legacy_components, st_result, components

				def evaluate_candidate(istep_big, closure_count):
					nonlocal best_accepted, best_violation
					with torch.no_grad():
						legacy_components, st_result, components = calculate_components()
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
					accepted = constraints_ok and condition_ok

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
					violation_key = (
						max(dft_violation, dpsi_violation, condition_violation),
						dft_violation + dpsi_violation + condition_violation,
					)
					if best_violation is None or violation_key < best_violation["key"]:
						best_violation = {
							"key": violation_key,
							"dft": dft_violation,
							"dpsi": dpsi_violation,
							"condition": condition_violation,
							"max_st_condition": st_result.max_condition,
							"condition_limit": loss_config["condition_limit"],
						}

					print(
						istep_big,
						closure_count,
						data_transmit["istep_all"],
						components["dft_origin"].item(),
						components["dft_dpsi"].item(),
						components["sternheimer"].item(),
						components["constraint_dft"].item(),
						components["constraint_dpsi"].item(),
						components["total"].item(),
						st_result.max_condition,
						str(accepted).lower(),
						sep="\t",
						file=files[1],
					)
					data_transmit["istep_all"] += 1
					data_transmit.update(
						{
							"Loss": components["total"].item(),
							"Spillage": components["total"].item(),
						}
					)

					if accepted and (
						best_accepted is None
						or components["sternheimer"].item()
						< best_accepted["loss_components"]["sternheimer"]
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
							"max_st_condition": st_result.max_condition,
						}
						data_transmit["flag_finish"] = 0
					else:
						data_transmit["flag_finish"] += 1

				def closure():
					opt.zero_grad()
					_, _, components = calculate_components()
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
					f"condition={best_violation['condition']:.8g}; "
					f"max_st_condition={best_violation['max_st_condition']:.8g}, "
					f"condition_limit={best_violation['condition_limit']:.8g}"
				)
			data_transmit.update(best_accepted)
			data_transmit["Loss"] = best_accepted["loss_components"]["total"]
			data_transmit["Spillage"] = best_accepted["loss_components"]["total"]
		return data_transmit
