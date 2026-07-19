import util
import torch
import numpy as np
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass


_LOSS_COMPONENTS = (
	("dft_origin", "DFT origin loss"),
	("dft_dpsi", "DFT dpsi loss"),
	("sternheimer", "Sternheimer loss"),
	("regularization_dpsi", "dpsi regularization loss"),
	("constraint_dft", "DFT constraint loss"),
	("constraint_dpsi", "dpsi constraint loss"),
	("total", "Total loss"),
)


@dataclass(frozen=True)
class CoefficientInitializationMetadata:
	loaded_indices: frozenset
	appended_indices: frozenset


def _validate_loss_metadata(loss_components, mode):
	if loss_components is None and mode is None:
		return None
	if loss_components is None or mode is None:
		raise ValueError("mode and loss_components must be supplied together")
	if mode not in ("st_only", "st_constrained", "st_dpsi_joint"):
		raise ValueError(f"invalid mode {mode!r}")
	if not isinstance(loss_components, Mapping):
		raise TypeError("loss_components must be a mapping")
	expected = {name for name, _ in _LOSS_COMPONENTS}
	if set(loss_components) != expected:
		raise ValueError(
			"loss_components must contain exactly "
			+ ", ".join(name for name, _ in _LOSS_COMPONENTS)
		)

	validated = {}
	for name, _ in _LOSS_COMPONENTS:
		value = loss_components[name]
		if (
			isinstance(value, bool)
			or not isinstance(value, numbers.Real)
			or not math.isfinite(value)
			or value < 0.0
		):
			raise ValueError(f"{name} metadata must be finite and nonnegative")
		validated[name] = float(value)
	return validated

def random_C_init(info_element):
	""" C[it][il][ie,iu]	<jY|\phi> """
	C = dict()
	for it in info_element.keys():
		C[it] = util.ND_list(info_element[it].Nl)
		for il in range(info_element[it].Nl):
			C[it][il] = torch.tensor(np.random.uniform(-1,1, (info_element[it].Ne, info_element[it].Nu[il])), dtype=torch.float64, requires_grad=True)
	return C



def read_C_init(file_name, info_element, return_metadata=False):
	""" C[it][il][ie,iu]	<jY|\phi> """
	if not isinstance(return_metadata, bool):
		raise TypeError("return_metadata must be a bool")
	C = random_C_init(info_element)

	with open(file_name,"r") as file:

		for line in file:
			if line.strip() == "<Coefficient>":
				line=None
				break
		util.ignore_line(file,1)

		C_read_index = set()
		while True:
			line = file.readline()
			if not line:
				raise IOError(
					"missing </Coefficient> in read_C_init " + file_name
				)
			line = line.strip()
			if not line:
				continue
			if line.startswith("Type"):
				label = file.readline().split()
				if len(label) != 3:
					raise IOError(
						"invalid coefficient label in read_C_init " + file_name
					)
				it, il_text, iu_text = label
				try:
					il = int(il_text)
					iu = int(iu_text) - 1
				except ValueError as exc:
					raise IOError(
						"invalid coefficient label in read_C_init " + file_name
					) from exc
				index = (it, il, iu)
				if (
					it not in info_element
					or il < 0
					or il >= info_element[it].Nl
					or iu < 0
					or iu >= info_element[it].Nu[il]
				):
					raise ValueError(
						f"coefficient column {index!r} is outside requested Nu"
					)
				if index in C_read_index:
					raise ValueError(f"duplicate coefficient column {index!r}")

				values = []
				while len(values) < info_element[it].Ne:
					value_line = file.readline()
					if not value_line:
						raise IOError(
							f"coefficient column {index!r} is incomplete"
						)
					fields = value_line.split()
					if not fields:
						continue
					if fields[0] == "Type" or fields[0] == "</Coefficient>":
						raise IOError(
							f"coefficient column {index!r} is incomplete"
						)
					try:
						values.extend(float(value) for value in fields)
					except ValueError as exc:
						raise IOError(
							f"invalid value in coefficient column {index!r}"
						) from exc
				if len(values) != info_element[it].Ne:
					raise IOError(
						f"coefficient column {index!r} has {len(values)} values, "
						f"expected {info_element[it].Ne}"
					)
				with torch.no_grad():
					C[it][il][:, iu] = torch.tensor(
						values, dtype=C[it][il].dtype
					)
				C_read_index.add(index)
			elif line.startswith("</Coefficient>"):
				break
			else:
				raise IOError("unknown line in read_C_init "+file_name+"\n"+line)

	requested_indices = frozenset(
		(it, il, iu)
		for it in info_element
		for il in range(info_element[it].Nl)
		for iu in range(info_element[it].Nu[il])
	)
	loaded_indices = frozenset(C_read_index)
	metadata = CoefficientInitializationMetadata(
		loaded_indices=loaded_indices,
		appended_indices=requested_indices - loaded_indices,
	)
	if return_metadata:
		return C, metadata
	return C, C_read_index



def copy_C(C,info_element):
	C_copy = dict()
	for it in info_element.keys():
		C_copy[it] = util.ND_list(info_element[it].Nl)
		for il in range(info_element[it].Nl):
			C_copy[it][il] = C[it][il].clone()
	return C_copy



def write_C(file_name,C,Spillage,loss_components=None,mode=None):
	loss_components = _validate_loss_metadata(loss_components, mode)
	with open(file_name,"w") as file:
		print("<Coefficient>", file=file)
		#print("\tTotal number of radial orbitals.", file=file)
		nTotal = 0
		for it,C_t in C.items():
			for il,C_tl in enumerate(C_t):
				for iu in range(C_tl.size()[1]):
					nTotal += 1
			#nTotal = sum(info["Nu"][it])
		print("\t %s Total number of radial orbitals."%nTotal , file=file)
		#print("\tTotal number of radial orbitals.", file=file)
		for it,C_t in C.items():
			for il,C_tl in enumerate(C_t):
				for iu in range(C_tl.size()[1]):
					print("\tType\tL\tZeta-Orbital", file=file)
					print(f"\t  {it} \t{il}\t    {iu+1}", file=file)
					for ie in range(C_tl.size()[0]):
						print("\t", '%18.14f'%C_tl[ie,iu].item(), file=file)
		print("</Coefficient>", file=file)
		print("<Mkb>", file=file)
		print("Left spillage = %.10e"%Spillage, file=file)
		if loss_components is not None:
			print(f"Mode = {mode}", file=file)
			for name, label in _LOSS_COMPONENTS:
				print(f"{label} = {loss_components[name]:.10e}", file=file)
		print("</Mkb>", file=file)



def cover_C(C_old, info_element):
	""" C[it][il][ie,iu] """
	C_new = random_C_init(info_element)
	C_read_index = set()
	for it in C_old:
		for il in range(len(C_old[it])):
			size = C_old[it][il].size()
			with torch.no_grad():
				C_new[it][il][:size[0],:size[1]] = C_old[it][il]
			C_read_index.update({(it,il,iu) for iu in range(size[1])})
	return C_new, C_read_index


#def init_C(info):
#	""" C[it][il][ie,iu] """
#	C = ND_list(max(info.Nt))
#	for it in range(len(C)):
#		C[it] = ND_list(info.Nl[it])
#		for il in range(info.Nl[it]):
#			C[it][il] = torch.autograd.Variable( torch.Tensor( info.Ne, info.Nu[it][il] ), requires_grad = True )
#
#	with open("C_init.dat","r") as file:
#		line = []
#		for it in range(len(C)):
#			for il in range(info.Nl[it]):
#				for i_n in range(info.Nu[it][il]):
#					for ie in range(info.Ne[it]):
#						if not line:	line=file.readline().split()
#						C[it][il].data[ie,i_n] = float(line.pop(0))
#	return C
