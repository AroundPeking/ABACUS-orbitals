import re
import torch
import itertools

# occ[ik][ib]
def read_istate(file_name):
	with open(file_name, "r") as file:
		content = file.read()
	if "Electronic state energy (eV) and occupations" in content:
		return read_eig_occ(content)
	return read_legacy_istate(file_name)


def read_eig_occ(content):
	ionic_steps = re.split(r"(?m)^\s*\d+\s+#\s*ionic step\s*$", content)
	content = next((step for step in reversed(ionic_steps) if "Spin number" in step), "")
	spin_match = re.search(r"Spin number\s+(\d+)", content)
	if not spin_match:
		raise ValueError("Missing 'Spin number' in eig_occ output")
	nspin0 = int(spin_match.group(1))
	if nspin0 not in (1, 2):
		raise ValueError("Unsupported spin count in eig_occ output: %s" % nspin0)

	blocks = re.split(r"(?m)^\s*spin=(\d+)\s+k-point=(\d+)/(\d+)[^\n]*$", content)
	if len(blocks) == 1:
		raise ValueError("Missing spin/k-point blocks in eig_occ output")

	occ = [[] for _ in range(nspin0)]
	expected_nk = None
	seen = set()
	for index in range(1, len(blocks), 4):
		ispin = int(blocks[index])
		ik = int(blocks[index + 1])
		nk = int(blocks[index + 2])
		block = blocks[index + 3]
		if not 1 <= ispin <= nspin0:
			raise ValueError("Invalid spin index in eig_occ output: %s" % ispin)
		if expected_nk is None:
			expected_nk = nk
		elif nk != expected_nk:
			raise ValueError("Inconsistent k-point count in eig_occ output")
		key = (ispin, ik)
		if key in seen:
			raise ValueError("Duplicate spin/k-point block in eig_occ output: %s" % (key,))
		seen.add(key)

		values = []
		for line in block.splitlines():
			fields = line.split()
			if len(fields) != 3:
				continue
			try:
				int(fields[0])
				values.append(float(fields[2]))
			except ValueError:
				continue
		if not values:
			raise ValueError("Missing occupations for spin %s k-point %s" % key)
		occ[ispin - 1].append((ik, torch.Tensor(values)))

	if expected_nk is None or len(seen) != nspin0 * expected_nk:
		raise ValueError("Incomplete spin/k-point blocks in eig_occ output")
	for ispin in range(nspin0):
		occ[ispin].sort(key=lambda item: item[0])
		if [ik for ik, _ in occ[ispin]] != list(range(1, expected_nk + 1)):
			raise ValueError("Incomplete k-point sequence in eig_occ output")
		occ[ispin] = [values for _, values in occ[ispin]]
	return list(itertools.chain(*occ))


def read_legacy_istate(file_name):
	nspin0 = get_nspin0(file_name)
	if nspin0==1:	occ = [[]]
	elif nspin0==2:	occ = [[],[]]
	with open(file_name,"r") as file:
		content = file.read().split("BAND")
		for content_k in content[1:]:
			content_k = content_k.split("\n")
			k = get_k(content_k[0])
			for ispin in range(nspin0):
				occ[ispin].append([])
			for line in content_k[1:]:
				line = line.strip()
				if line:
					line = line.split()
					if nspin0==1:
						occ[0][-1].append(float(line[2]))
					elif nspin0==2:
						occ[0][-1].append(float(line[2]))
						occ[1][-1].append(float(line[4]))
			for ispin in range(nspin0):
				occ[ispin][-1] = torch.Tensor(occ[ispin][-1])
	occ = list(itertools.chain(*occ))
	return occ

def get_k(line):
	k = re.compile(r"Kpoint\s*=\s*(\d+)").search(line).group(1)
	return int(k)

def get_nspin0(file_name):
	with open(file_name,"r") as file:
		file.readline()
		line = file.readline()
		lens = len(line.split())
		if lens == 3:	return 1
		elif lens == 5:	return 2
		else:	raise
