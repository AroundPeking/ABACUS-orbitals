#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import IO.read_QSV
import IO.print_QSV
import IO.func_C
import IO.read_json
import IO.read_sternheimer
import IO.read_sternheimer_source
import IO.print_orbital
import IO.cal_weight
import IO.change_info
import orbital
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from attribute_dict import AttributeDict
from opt_orbital_converge import Opt_Orbital_Converge
from freeze_orbitals import validate_freeze_orbitals
from IO.read_zero_order_audit import read_zero_order_audit
from optimization_loss import normalize_loss_config
from projected_pi_optimization import (
	NormalizedPhysicalFamilyProjectedPiOptimization,
)
from radial_locality import RadialSubspaceLocality
from response_family_spillage import NormalizedPhysicalFamilySpillage
from response_selection import ResponseTargetFamily
from sternheimer_spillage import OrbitalColumn
from sternheimer_source_pair import pair_response_and_source
from sternheimer_targets import apply_target_element_aliases, parse_target_entries

import numpy as np
import torch
import time
import pprint
import sys


@dataclass(frozen=True)
class LoadedSternheimerTargets:
	entries: tuple
	families: tuple
	projected_pi_pairs: tuple = ()
	zero_order_audits: tuple = ()


_PROJECTED_PI_MODES = frozenset(
	{"pi_dpsi_joint", "pi_rpa_sensitive_joint"}
)


def _sha256(path):
	digest = hashlib.sha256()
	with Path(path).open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _write_projected_pi_metadata(path, targets, data_transmit):
	diagnostics = data_transmit["projected_pi_diagnostics"]
	pairs = dict(targets.projected_pi_pairs)
	audits = dict(targets.zero_order_audits)
	entries = {entry.family: entry for entry in targets.entries}
	inputs = {}
	for family in ("H", "H2"):
		entry = entries[family]
		audit = audits[family]
		pair = pairs[family]
		inputs[family] = {
			"response_path": str(entry.path),
			"response_sha256": _sha256(entry.path),
			"source_path": str(entry.source_path),
			"source_sha256": _sha256(entry.source_path),
			"zero_order_audit_path": str(entry.zero_order_audit_path),
			"zero_order_audit_sha256": _sha256(
				entry.zero_order_audit_path
			),
			"zero_order_identity": {
				"passed": audit.passed,
				"occupied_state_count": audit.occupied_state_count,
				"grid": list(audit.grid),
				"max_occupation_abs_diff": audit.max_occupation_abs_diff,
				"max_occupied_eigenvalue_abs_diff_ha": (
					audit.max_occupied_eigenvalue_abs_diff_ha
				),
				"final_total_energy_abs_diff_ha": (
					audit.final_total_energy_abs_diff_ha
				),
				"source_file_sha256": dict(audit.source_file_sha256),
			},
			"response_provenance": pair.response.provenance,
			"source_provenance": pair.source.provenance,
			"provenance_warnings": list(pair.provenance_warnings),
		}
	payload = {
		"schema_version": 1,
		"mode": data_transmit.get("loss_mode", "pi_dpsi_joint"),
		"loss_components": data_transmit["loss_components"],
		"projected_pi": diagnostics,
		"inputs": inputs,
		"uses_sos_energy": False,
		"uses_ghost_family": False,
	}
	Path(path).write_text(
		json.dumps(payload, indent=2, sort_keys=True) + "\n",
		encoding="utf-8",
	)


def _load_sternheimer_data(file_list, info_optimize):
	stages = [
		normalize_loss_config(stage["loss"])
		for stage in info_optimize
		if "loss" in stage
	]
	modes = {stage["mode"] for stage in stages}
	projected_pi_modes = modes & _PROJECTED_PI_MODES
	projected_pi_mode = bool(projected_pi_modes)
	if projected_pi_mode and (
		len(projected_pi_modes) != 1 or modes != projected_pi_modes
	):
		mode = (
			"pi_rpa_sensitive_joint"
			if "pi_rpa_sensitive_joint" in projected_pi_modes
			else "pi_dpsi_joint"
		)
		if mode == "pi_dpsi_joint":
			raise ValueError(
				"cannot mix pi_dpsi_joint and legacy Sternheimer loss stages"
			)
		raise ValueError(
			"cannot mix pi_rpa_sensitive_joint and other loss stages"
		)
	projected_pi_label = (
		next(iter(projected_pi_modes)) if projected_pi_mode else None
	)
	if "sternheimer" in file_list:
		values = file_list["sternheimer"]
		if not isinstance(values, (list, tuple)) or not values:
			raise ValueError("SIAB Sternheimer targets require a nonempty list")
		entries = parse_target_entries(values)
		if not stages:
			raise ValueError(
				"sternheimer data requires a Sternheimer loss stage"
			)
		if projected_pi_mode:
			if any(entry.role == "ghost" for entry in entries):
				raise ValueError(
					f"{projected_pi_label} cannot consume ghost Sternheimer targets"
				)
			for entry in entries:
				if entry.source_path is None:
					raise ValueError(
						f"{projected_pi_label} target {entry.family} requires source_path"
					)
				if entry.zero_order_audit_path is None:
					raise ValueError(
						f"{projected_pi_label} target "
						f"{entry.family} requires zero_order_audit_path"
					)
			family_names = [entry.family for entry in entries]
			if (
				len(entries) != 2
				or len(set(family_names)) != 2
				or set(family_names) != {"H", "H2"}
			):
				raise ValueError(
					f"{projected_pi_label} requires exactly one H and one H2 target"
				)

			response_by_family = {}
			pairs = []
			audits = []
			entry_by_family = {entry.family: entry for entry in entries}
			for family in ("H", "H2"):
				entry = entry_by_family[family]
				response = apply_target_element_aliases(
					IO.read_sternheimer.read_sternheimer(entry.path), entry
				)
				source = apply_target_element_aliases(
					IO.read_sternheimer_source.read_sternheimer_source(
						entry.source_path
					),
					entry,
				)
				pair = pair_response_and_source(response, source)
				audit = read_zero_order_audit(
					entry.zero_order_audit_path, family
				)
				response_by_family[family] = response
				pairs.append((family, pair))
				audits.append((family, audit))
			families = tuple(
				ResponseTargetFamily(name, (response_by_family[name],), "physical")
				for name in ("H", "H2")
			)
			return LoadedSternheimerTargets(
				entries,
				families,
				tuple(pairs),
				tuple(audits),
			), stages

		physical = {}
		for entry in entries:
			if entry.role != "physical":
				continue
			physical.setdefault(entry.family, []).append(
				apply_target_element_aliases(
					IO.read_sternheimer.read_sternheimer(entry.path), entry
				)
			)
		if not physical:
			raise ValueError(
				"SIAB optimization requires a physical Sternheimer target"
			)
		families = tuple(
			ResponseTargetFamily(name, tuple(data), "physical")
			for name, data in physical.items()
		)
		return LoadedSternheimerTargets(entries, families), stages
	if stages:
		raise ValueError("Sternheimer loss stage requires sternheimer data")
	return None, stages


def _expand_fixed_orbitals(data, C, freeze_specs):
	freeze_indices = validate_freeze_orbitals(freeze_specs, C)
	if not freeze_indices:
		raise ValueError(
			"the first SIAB Sternheimer implementation requires nonempty freeze_orbitals"
		)

	fixed_orbitals = []
	seen = set()
	for spec in freeze_specs:
		element, l, zeta = spec["element"], int(spec["l"]), int(spec["zeta"])
		matching_blocks = [
			block
			for block in data.blocks
			if block.element == element and block.l == l
		]
		if not matching_blocks:
			raise ValueError(
				f"freeze orbital {(element, l, zeta)!r} maps to no primitive blocks"
			)
		for block in matching_blocks:
			column = OrbitalColumn(
				block.element, block.atom_index, block.l, block.m, zeta
			)
			if column in seen:
				raise ValueError(f"duplicate fixed orbital expansion {column!r}")
			seen.add(column)
			fixed_orbitals.append(column)
	return tuple(fixed_orbitals)


def _sternheimer_info_element(data, info_true):
	elements = tuple(info_true.Nt_all)
	if len(set(elements)) != len(elements):
		raise ValueError("element.Nt_all must not contain duplicates")
	if set(info_true.Nu) != set(elements):
		raise ValueError("element.Nu keys must match element.Nt_all")

	data_elements = {block.element for block in data.blocks}
	if data_elements != set(elements):
		raise ValueError(
			"Sternheimer primitive elements do not match element.Nt_all: "
			f"target={sorted(data_elements)!r}, input={sorted(elements)!r}"
		)

	info_element = AttributeDict()
	for element_index, element in enumerate(elements):
		nu = list(info_true.Nu[element])
		if (
			not nu
			or any(type(value) is not int or value < 0 for value in nu)
			or not any(value > 0 for value in nu)
		):
			raise ValueError(
				f"element.Nu[{element!r}] must contain nonnegative integers "
				"and at least one orbital"
			)
		nprimitive_by_l = []
		for l in range(len(nu)):
			counts = {
				block.n_primitive
				for block in data.blocks
				if block.element == element and block.l == l
			}
			if len(counts) != 1:
				raise ValueError(
					"Sternheimer target must define one primitive count for "
					f"every requested element/l; {element}/{l} has {sorted(counts)!r}"
				)
			nprimitive_by_l.append(counts.pop())
		if len(set(nprimitive_by_l)) != 1:
			raise ValueError(
				"SIAB currently requires one radial primitive count per element; "
				f"{element} has {nprimitive_by_l!r}"
			)

		info_element[element].index = element_index
		info_element[element].Nu = nu
		info_element[element].Nl = len(nu)
		info_element[element].Ne = nprimitive_by_l[0]
	return info_element


def _set_random_seed(info_C_init):
	if "seed" in info_C_init:
		seed = info_C_init["seed"]
		if type(seed) is not int:
			raise TypeError("seed must be a non-bool integer")
		if seed < 0 or seed >= 2**32:
			raise ValueError("seed must satisfy 0 <= seed < 2**32")
	else:
		seed = int(1000 * time.time()) % (2**32)
	np.random.seed(seed)
	torch.manual_seed(seed)
	print("numpy seed:", seed)
	print("torch seed:", seed)
	return seed


def _normalize_initial_coefficients(
	C, info_element, info_radial, E, freeze_specs=None
):
	frozen_columns = {}
	if freeze_specs is not None:
		freeze_indices = validate_freeze_orbitals(freeze_specs, C)
		frozen_columns = {
			index: C[index[0]][index[1]][:, index[2]].detach().clone()
			for index in freeze_indices
		}

	orbital.normalize(
		orbital.generate_orbital(info_element, info_radial, C, E),
		info_radial["dr"],
		C,
		flag_norm_C=True,
	)

	if frozen_columns:
		with torch.no_grad():
			for (element, l, zeta), column in frozen_columns.items():
				C[element][l][:, zeta].copy_(column)


def main():
	time_start = time.time()

	file_list, info_true, info_weight, info_optimize, info_C_init, info_V, info_radial = IO.read_json.read_json("INPUT")
	_set_random_seed(info_C_init)
	sternheimer_targets, sternheimer_stages = _load_sternheimer_data(
		file_list, info_optimize
	)
	sternheimer_data = (
		sternheimer_targets.families[0].data[0]
		if sternheimer_targets is not None
		else None
	)
	has_legacy_origin = "origin" in file_list
	projected_pi_modes = {
		stage["mode"]
		for stage in sternheimer_stages
		if stage["mode"] in _PROJECTED_PI_MODES
	}
	uses_projected_pi = bool(projected_pi_modes)
	projected_pi_mode = (
		next(iter(projected_pi_modes)) if uses_projected_pi else None
	)
	if not has_legacy_origin:
		if sternheimer_data is None or not sternheimer_stages:
			raise ValueError("SIAB input without origin requires Sternheimer data")
		if any(stage["mode"] != "st_only" for stage in sternheimer_stages):
			if uses_projected_pi:
				raise ValueError(
					f"{projected_pi_mode} requires origin and dpsi data"
				)
			raise ValueError(
				"st_constrained and st_dpsi_joint require origin and dpsi data"
			)
		if "linear" in file_list:
			raise ValueError("linear data requires origin data")
		info_kst = None
		info_stru = []
		info_element = _sternheimer_info_element(sternheimer_data, info_true)
	else:
		joint_modes = {
			stage["mode"]
			for stage in sternheimer_stages
			if stage["mode"]
			in ("st_dpsi_joint", "pi_dpsi_joint", "pi_rpa_sensitive_joint")
		}
		if joint_modes and "linear" not in file_list:
			mode = sorted(joint_modes)[0]
			raise ValueError(f"{mode} requires linear dpsi data")
		weight = IO.cal_weight.cal_weight(
			info_weight, info_V["same_band"], file_list["origin"]
		)
		info_kst = IO.read_QSV.read_file_head(info_true, file_list["origin"])
		info_stru, info_element = IO.change_info.change_info(
			info_kst, weight, info_V["same_band"]
		)
	#info_max = IO.change_info.get_info_max(info_stru, info_element)

	if info_kst is not None:
		print("info_kst:", pprint.pformat(info_kst), sep="\n", end="\n"*2)
	print("info_element:", pprint.pformat(info_element,width=40), sep="\n", end="\n"*2)
	print("info_optimize:", pprint.pformat(info_optimize,width=40), sep="\n", end="\n"*2)
	print("info_radial:", pprint.pformat(info_radial,width=40), sep="\n", end="\n"*2)
	print("info_stru:", pprint.pformat(info_stru), sep="\n", end="\n"*2)
	#print("info_max:", pprint.pformat(info_max), sep="\n", end="\n"*2)

	if has_legacy_origin:
		QI,SI,VI_origin = IO.read_QSV.read_QSV(
			info_stru, info_element, file_list["origin"], info_V
		)
		if "linear" in file_list.keys():
			QI_linear, SI_linear, VI_linear = list(zip(*( IO.read_QSV.read_QSV(info_stru, info_element, file, info_V) for file in file_list["linear"] )))

	if info_C_init["init_from_file"]:
		C, initialization = IO.func_C.read_C_init(
			info_C_init["C_init_file"], info_element, return_metadata=True
		)
		C_read_index = set(initialization.loaded_indices)
		def format_indices(indices):
			return [
				f"{element}/l{l}/zeta{zeta + 1}"
				for element, l, zeta in sorted(indices)
			]
		print("loaded coefficient columns:", format_indices(initialization.loaded_indices))
		print("appended response columns:", format_indices(initialization.appended_indices))
	else:
		C = IO.func_C.random_C_init(info_element)
	E = orbital.set_E(info_element, info_radial["Rcut"])
	freeze_specs = (
		info_C_init["freeze_orbitals"]
		if "freeze_orbitals" in info_C_init
		else None
	)
	_normalize_initial_coefficients(
		C, info_element, info_radial, E, freeze_specs
	)
	radial_locality = None
	locality_stages = [
		stage
		for stage in sternheimer_stages
		if stage["radial_tail_radius"] > 0.0
	]
	if locality_stages:
		if not freeze_specs:
			raise ValueError(
				"radial locality requires nonempty freeze_orbitals"
			)
		contracts = {
			(
				stage["radial_tail_radius"],
				stage["radial_tail_condition_limit"],
			)
			for stage in locality_stages
		}
		if len(contracts) != 1:
			raise ValueError(
				"all locality-enabled optimization stages must use the same "
				"radial_tail_radius and radial_tail_condition_limit"
			)
		local_radius, locality_condition_limit = contracts.pop()
		radial_locality = RadialSubspaceLocality(
			info_element,
			info_radial,
			E,
			freeze_specs,
			local_radius,
			condition_limit=locality_condition_limit,
		)

	sternheimer_spillage = None
	projected_pi_objective = None
	if sternheimer_targets is not None:
		freeze_specs = info_C_init.get("freeze_orbitals")
		if not freeze_specs:
			raise ValueError(
				"the first SIAB Sternheimer implementation requires nonempty freeze_orbitals"
			)
		condition_limit = max(
			stage["condition_limit"] for stage in sternheimer_stages
		)
		if uses_projected_pi:
			rank_tolerances = {
				stage["projected_pi_rank_tolerance"]
				for stage in sternheimer_stages
			}
			if len(rank_tolerances) != 1:
				raise ValueError(
					f"all {projected_pi_mode} stages must use one "
					"projected_pi_rank_tolerance"
				)
			rank_tolerance = rank_tolerances.pop()
			if projected_pi_mode == "pi_dpsi_joint":
				projected_pi_objective = (
					NormalizedPhysicalFamilyProjectedPiOptimization(
						*sternheimer_targets.projected_pi_pairs,
						relative_rank_tolerance=rank_tolerance,
						condition_limit=condition_limit,
					)
				)
			else:
				sensitivity_alphas = {
					stage["projected_pi_sensitivity_alpha"]
					for stage in sternheimer_stages
				}
				if len(sensitivity_alphas) != 1:
					raise ValueError(
						"all pi_rpa_sensitive_joint stages must use one "
						"projected_pi_sensitivity_alpha"
					)
				projected_pi_objective = (
					NormalizedPhysicalFamilyProjectedPiOptimization(
						*sternheimer_targets.projected_pi_pairs,
						relative_rank_tolerance=rank_tolerance,
						condition_limit=condition_limit,
						sensitivity_alpha=sensitivity_alphas.pop(),
						family_power=4,
					)
				)
			for family, pair in sternheimer_targets.projected_pi_pairs:
				for warning in pair.provenance_warnings:
					print(f"projected-Pi {family} warning: {warning}")
		else:
			sternheimer_spillage = NormalizedPhysicalFamilySpillage(
				sternheimer_targets.families,
				C,
				C,
				freeze_specs,
				condition_limit=condition_limit,
			)

	opt_orb_conv = Opt_Orbital_Converge()
	opt_orb_conv.set_info(file_list, info_optimize, info_stru, info_C_init, info_V)
	opt_orb_conv.set_info_element(info_element)
	if has_legacy_origin:
		opt_orb_conv.set_QSVI(QI, SI, VI_origin)
	if sternheimer_spillage is not None:
		opt_orb_conv.set_sternheimer_spillage(sternheimer_spillage)
	if projected_pi_objective is not None:
		opt_orb_conv.set_projected_pi_objective(projected_pi_objective)
	if radial_locality is not None:
		opt_orb_conv.set_radial_locality(radial_locality)
	if "linear" in file_list.keys():
		opt_orb_conv.set_QSVI_linear(QI_linear, SI_linear, VI_linear)
	if info_C_init["init_from_file"]:
		opt_orb_conv.set_C_read_index(C_read_index)
	opt_orb_conv.set_E(E)

	with open("Spillage.dat","w") as S_file:
		data_transmit = opt_orb_conv.cal_converge(C, (sys.stdout,S_file))

	#orbital.normalize(
	#	orbital.generate_orbital(info_element, info_radial, C, E),
	#	{it:info_element[it].dr for it in info_element},
	#	C, flag_norm_C=True)

	orb = orbital.generate_orbital(info_element, info_radial, data_transmit["C"], E)
	for it in info_element:
		if info_radial["smearing_sigma"][it]:
			orbital.smooth_orbital(
				orb[it],
				info_radial["Rcut"][it],
				info_radial["dr"][it],
				info_radial["smearing_sigma"][it])
		orbital.orth(
			orb[it],
			info_radial["dr"][it])
	IO.print_orbital.print_orbital(
		orb,
		info_radial)
	IO.print_orbital.plot_orbital(
		orb,
		info_radial["Rcut"],
		info_radial["dr"])

	loss_diagnostics = None
	if "loss_components" in data_transmit:
		if data_transmit.get("loss_mode") in _PROJECTED_PI_MODES:
			pi_diagnostics = data_transmit["projected_pi_diagnostics"]
			loss_diagnostics = {
				"max_projected_pi_condition": data_transmit[
					"max_projected_pi_condition"
				],
				"lowest_projected_pi_frequency_ha": pi_diagnostics[
					"lowest_frequency_ha"
				],
				"lowest_projected_pi_loss": pi_diagnostics[
					"lowest_frequency_loss"
				],
				"projected_pi_rank_tolerance": pi_diagnostics[
					"rank_tolerance"
				],
			}
		else:
			loss_diagnostics = {
				"max_st_condition": data_transmit["max_st_condition"],
				"max_locality_condition": data_transmit[
					"max_locality_condition"
				],
			}
			loss_diagnostics.update(
				data_transmit.get("low_frequency_diagnostics", {})
			)

	IO.func_C.write_C(
		"ORBITAL_RESULTS.txt",
		data_transmit["C"],
		data_transmit["Spillage"],
		loss_components=data_transmit.get("loss_components"),
		mode=data_transmit.get("loss_mode"),
		diagnostics=loss_diagnostics,
	)
	if data_transmit.get("loss_mode") in _PROJECTED_PI_MODES:
		_write_projected_pi_metadata(
			"PROJECTED_PI_METADATA.json",
			sternheimer_targets,
			data_transmit,
		)

	print("Time (PyTorch):     %s\n"%(time.time()-time_start) )


if __name__=="__main__":
	np.set_printoptions(threshold=sys.maxsize, linewidth=10000)
	print( sys.version )
	main()
