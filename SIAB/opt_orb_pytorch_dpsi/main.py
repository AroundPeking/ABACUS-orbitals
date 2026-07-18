#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import IO.read_QSV
import IO.print_QSV
import IO.func_C
import IO.read_json
import IO.read_sternheimer
import IO.print_orbital
import IO.cal_weight
import IO.change_info
import orbital
import addict
from opt_orbital_converge import Opt_Orbital_Converge
from freeze_orbitals import validate_freeze_orbitals
from optimization_loss import normalize_loss_config
from sternheimer_spillage import OrbitalColumn, SternheimerSpillage

import numpy as np
import torch
import time
import pprint
import sys


def _load_sternheimer_data(file_list, info_optimize):
	stages = [
		normalize_loss_config(stage["loss"])
		for stage in info_optimize
		if "loss" in stage
	]
	if "sternheimer" in file_list:
		paths = file_list["sternheimer"]
		if not isinstance(paths, (list, tuple)) or len(paths) != 1:
			raise ValueError(
				"the first SIAB Sternheimer implementation requires exactly one data file"
			)
		if not stages:
			raise ValueError(
				"sternheimer data requires a Sternheimer loss stage"
			)
		return IO.read_sternheimer.read_sternheimer(paths[0]), stages
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

	info_element = addict.Dict()
	for element_index, element in enumerate(elements):
		nu = list(info_true.Nu[element])
		if not nu or any(type(value) is not int or value <= 0 for value in nu):
			raise ValueError(
				f"element.Nu[{element!r}] must contain positive integers"
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
	sternheimer_data, sternheimer_stages = _load_sternheimer_data(
		file_list, info_optimize
	)
	has_legacy_origin = "origin" in file_list
	if not has_legacy_origin:
		if sternheimer_data is None or not sternheimer_stages:
			raise ValueError("SIAB input without origin requires Sternheimer data")
		if any(stage["mode"] != "st_only" for stage in sternheimer_stages):
			raise ValueError("st_constrained requires origin and dpsi data")
		if "linear" in file_list:
			raise ValueError("linear data requires origin data")
		info_kst = None
		info_stru = []
		info_element = _sternheimer_info_element(sternheimer_data, info_true)
	else:
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
		C, C_read_index = IO.func_C.read_C_init( info_C_init["C_init_file"], info_element )
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

	sternheimer_spillage = None
	if sternheimer_data is not None:
		freeze_specs = info_C_init.get("freeze_orbitals")
		if not freeze_specs:
			raise ValueError(
				"the first SIAB Sternheimer implementation requires nonempty freeze_orbitals"
			)
		fixed_orbitals = _expand_fixed_orbitals(
			sternheimer_data, C, freeze_specs
		)
		condition_limit = max(
			stage["condition_limit"] for stage in sternheimer_stages
		)
		sternheimer_spillage = SternheimerSpillage(
			sternheimer_data,
			C,
			fixed_orbitals,
			condition_limit=condition_limit,
		)

	opt_orb_conv = Opt_Orbital_Converge()
	opt_orb_conv.set_info(file_list, info_optimize, info_stru, info_C_init, info_V)
	opt_orb_conv.set_info_element(info_element)
	if has_legacy_origin:
		opt_orb_conv.set_QSVI(QI, SI, VI_origin)
	if sternheimer_spillage is not None:
		opt_orb_conv.set_sternheimer_spillage(sternheimer_spillage)
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

	IO.func_C.write_C(
		"ORBITAL_RESULTS.txt",
		data_transmit["C"],
		data_transmit["Spillage"],
		loss_components=data_transmit.get("loss_components"),
		mode=data_transmit.get("loss_mode"),
	)

	print("Time (PyTorch):     %s\n"%(time.time()-time_start) )


if __name__=="__main__":
	np.set_printoptions(threshold=sys.maxsize, linewidth=10000)
	print( sys.version )
	main()
