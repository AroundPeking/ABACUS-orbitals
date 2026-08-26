import math
from pathlib import Path
import re


KPOINT_HEADER = re.compile(
    r"^spin=(?P<spin>\d+)\s+k-point=(?P<index>\d+)/(?P<total>\d+)\b"
)


def read_occupied_band_count(path, occupation_threshold=1.0e-12):
    path = Path(path)
    if occupation_threshold <= 0.0 or not math.isfinite(occupation_threshold):
        raise ValueError("occupation threshold must be finite and positive")

    spin_number = None
    expected_kpoints = None
    current = None
    records = {}
    for raw_line in path.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if line.startswith("Spin number "):
            spin_number = int(line.split()[-1])
            continue
        header = KPOINT_HEADER.match(line)
        if header:
            spin = int(header.group("spin"))
            index = int(header.group("index"))
            total = int(header.group("total"))
            if expected_kpoints is None:
                expected_kpoints = total
            if total != expected_kpoints or index < 1 or index > total:
                raise ValueError("eig_occ contains inconsistent k-point headers")
            current = (spin, index)
            if current in records:
                raise ValueError("eig_occ contains a duplicate spin/k-point record")
            records[current] = []
            continue
        if current is None:
            continue
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            band = int(fields[0])
            energy = float(fields[1])
            occupation = float(fields[2])
        except ValueError:
            continue
        if band != len(records[current]) + 1:
            raise ValueError("eig_occ band indices must be consecutive")
        if not math.isfinite(energy) or not math.isfinite(occupation):
            raise ValueError("eig_occ contains non-finite band data")
        if occupation < -occupation_threshold:
            raise ValueError("eig_occ contains a negative occupation")
        records[current].append(occupation)

    if spin_number != 1:
        raise ValueError("head/wing input requires one spin channel")
    if expected_kpoints is None or len(records) != expected_kpoints:
        raise ValueError("eig_occ does not contain every irreducible k point")
    if any(spin != 1 for spin, _ in records):
        raise ValueError("eig_occ contains an unexpected spin channel")

    occupied_counts = []
    for occupations in records.values():
        occupied = [value > occupation_threshold for value in occupations]
        count = sum(occupied)
        if count <= 0 or occupied != [True] * count + [False] * (len(occupied) - count):
            raise ValueError("eig_occ occupied bands must form a nonempty prefix")
        occupied_counts.append(count)
    if len(set(occupied_counts)) != 1:
        raise ValueError("every k point must have the same occupied band count")
    return occupied_counts[0]


def get_omega(filename : str = 'LibRPA_freq.out'):

    with open(filename, 'r') as file:
        lines = file.readlines()

    # 提取 "Frequency node & weight:" 和 "Time node & weight:" 之间的值
    freq_values = []
    extract = False
    for line in lines:
        if 'Frequency node & weight:' in line:
            extract = True
            continue
        if 'Time node & weight:' in line:
            extract = False
        if extract:
            # 提取第二列的值
            data = line.split()
            if len(data) > 1:
                freq_values.append(float(data[1]))

    # 将 freq_values 中的值从 Hartree 转换为 eV
    omega_values = [value * 27.2114079527 for value in freq_values]

    return omega_values

def get_param(work_dir : str = './'):
    '''
    get lattice_vector from STRU,
    get fermi_energy(eV) from running_scf.log,
    get occ_band from the completed ABACUS eig_occ.txt.
    '''
    import os
    f_stru = os.path.join(work_dir, 'STRU')
    f_running = os.path.join(work_dir, "OUT.ABACUS/running_scf.log")
    f_eig_occ = os.path.join(work_dir, 'OUT.ABACUS/eig_occ.txt')

    lattice_vector = []
    with open(f_stru, 'r') as file:
        lines = file.readlines()
        # 找到 "LATTICE_VECTORS" 行
        start_index = 0
        for i, line in enumerate(lines):
            if line.strip() == "LATTICE_VECTORS":
                start_index = i + 1  # "LATTICE_VECTORS" 行的下一行是第一个向量
                break
        # 读取接下来的三行作为 LATTICE_VECTORS
        for i in range(start_index, start_index + 3):
            vector = list(map(float, lines[i].split()))
            lattice_vector.append(vector)

    fermi_energy = None
    with open(f_running, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue

            if parts[0].upper() == 'E_FERMI':
                numeric_values = []
                for part in parts[1:]:
                    try:
                        numeric_values.append(float(part))
                    except ValueError:
                        continue
                if numeric_values:
                    fermi_energy = numeric_values[-1]
                continue

            if 'EFERMI' in parts:
                for i, part in enumerate(parts):
                    if part == 'EFERMI' and i + 2 < len(parts):
                        try:
                            fermi_energy = float(parts[i + 2])
                        except ValueError:
                            pass
                        break

    if fermi_energy is None:
        raise ValueError(f"Failed to find Fermi energy in {f_running}")

    occ_band = read_occupied_band_count(f_eig_occ)
    return lattice_vector, fermi_energy, occ_band

def dat2out():
    import numpy as np
    # 读取dielectric_function_real.dat文件
    with open('dielectric_function_real.dat', 'r') as real_file:
        real_lines = real_file.readlines()[1:]

    # 读取dielectric_function_imag.dat文件
    with open('dielectric_function_imag.dat', 'r') as imag_file:
        imag_lines = imag_file.readlines()[1:]

    # 创建新文件dielecfunc_out
    with open('dielecfunc_out', 'w') as output_file:
        # 写入文件头部
        #output_file.write("Real_Column_1 Real_Column_2 Imag_Column_2\n")

        # 将数据写入新文件
        for real_line, imag_line in zip(real_lines, imag_lines):
            freq_data = [float(value)/27.2114079527 for value in real_line.split()[0:1]]
            xx_data = [float(value) for value in real_line.split()[1:2]]
            yy_data = [float(value) for value in real_line.split()[5:6]]
            zz_data = [float(value) for value in real_line.split()[9:10]]
            real_data = ((np.array(xx_data) + np.array(yy_data) + np.array(zz_data))/3.0).tolist()
            imag_data = [float(value) for value in imag_line.split()[1:2]]
            #print(real_line.split()[0:2])
            output_line = "{:.8f} {:.8f} {:.8f}\n".format(*freq_data, *real_data, *imag_data)
            output_file.write(output_line)

def read_KPT(file_path='./KPT'):
    with open(file_path, 'r') as file:
        for line in file:
            # 尝试将行内容按空格分割并转换为整数
            try:
                numbers = list(map(int, line.strip().split()))
                if len(numbers) == 6:
                    #if numbers[0] == numbers[1] and numbers[1] == numbers[2]:
                    #    return numbers[0]
                    #else:
                    #    return None
                    return numbers[0], numbers[1], numbers[2]
            except ValueError:
                # 如果转换失败，跳过该行
                continue
    return None  # 如果没有找到符合条件的行，返回None

if __name__ == '__main__':
    # ------------ set inputs here
    # when soc nspin=4
    abacus_dir = './'
    nspin = 1
    use_soc = False
    kpt_file_dir = './KPT_scf'
    # ------------ over
    nkx, nky, nkz = read_KPT(kpt_file_dir)
    #print(nkx, nky, nkz)
    import output_librpa
    lattice_vector, fermi_energy, occ_band = get_param(abacus_dir)
    output_librpa.output_librpa(lattice_vector, fermi_energy, occ_band, nkx=nkx, nky=nky, nkz=nkz, nspin=nspin, use_soc=use_soc)
