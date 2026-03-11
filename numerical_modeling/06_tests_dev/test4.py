# -*- coding: utf-8 -*-
# make_hetero_from_inp_cdp_2minerals_with_disks.py

from abaqus import *
from abaqusConstants import *
import regionToolset
import random
import math
import os
import sys

# ============================================================
# 1) USER PARAMETERS
# ============================================================

MODEL_NAME = "Model-1"

DEFAULT_INP = "test_v10.inp"
DEFAULT_OUT = "test_v10_hetero.inp"

# If Abaqus picks the wrong part, force it here (e.g. "Cyl3D")
FORCE_PART_NAME = None  # "Cyl3D"

# ---- Disks settings ----
DISK_PART_NAME = "Disque"
DISK_INST_BOTTOM = "Disque-1"
DISK_INST_TOP    = "Disque-2"
DISK_GAP = 0.0  # small gap between disk and specimen (same unit as model)

RANDOM_SEED = 42
a = 3.0
Nwaves = 300
MIN1_IS_LOW = True

# Two-phase volume fractions (must sum to 1.0)
p_frac = (0.50, 0.50)

# Minerals 1 & 2 only
minerals = [
    dict(name="Mineral-1", E=35000.0, nu=0.25, fc0=120.0, ft0=5.0),
    dict(name="Mineral-2", E=72000.0, nu=0.25, fc0=285.0, ft0=11.5),
]

ELSET_NAMES = ["ELSET_MIN1", "ELSET_MIN2"]
SEC_NAMES   = ["SEC_MIN1",   "SEC_MIN2"]

# CDP constants (same as your rock)
CDP_DILAT = 35.0
CDP_ECC   = 0.1
CDP_FBFC  = 1.16
CDP_K     = 0.554
CDP_VISC  = 5e-05

# Rock-based reference curves (your rock M0)
_R_F1     = 153.7 / 196.4
_R_CRES   = 4.27  / 196.4
_EIN_PIC  = 0.00065
_EIN_END  = 0.0094

_R_TRES   = 0.034 / 8.78
_ETIN_END = 0.00035

DC_POINTS = [(0.0, 0.0)]
DT_POINTS = [(0.0, 0.0), (0.98, _ETIN_END)]

# ============================================================
# 2) SMALL HELPERS
# ============================================================

def safe_del(container, key):
    try:
        if key in container:
            del container[key]
    except Exception:
        pass

def element_volume(part, elem):
    try:
        return float(elem.getSize())
    except Exception:
        xs, ys, zs = [], [], []
        for nl in elem.connectivity:
            n = part.nodes[nl - 1]
            xs.append(n.coordinates[0]); ys.append(n.coordinates[1]); zs.append(n.coordinates[2])
        return (max(xs)-min(xs))*(max(ys)-min(ys))*(max(zs)-min(zs))

def node_xyz(part, node_label):
    n = part.nodes[node_label - 1]
    return n.coordinates[0], n.coordinates[1], n.coordinates[2]

def pick_target_part(mdl):
    if FORCE_PART_NAME is not None:
        if FORCE_PART_NAME not in mdl.parts:
            raise RuntimeError("FORCE_PART_NAME '%s' not found in mdl.parts." % FORCE_PART_NAME)
        return mdl.parts[FORCE_PART_NAME]

    best = None
    best_ne = -1
    for pname, p in mdl.parts.items():
        try:
            if p.dimensionality != THREE_D:
                continue
            if p.type != DEFORMABLE_BODY:
                continue
        except Exception:
            continue
        ne = len(p.elements)
        if ne > best_ne:
            best_ne = ne
            best = p

    if best is None or best_ne <= 0:
        for pname, p in mdl.parts.items():
            ne = len(p.elements)
            if ne > best_ne:
                best_ne = ne
                best = p
        if best is None or best_ne <= 0:
            raise RuntimeError("Could not auto-detect a meshed part.")
    return best

def get_base_material_from_part(mdl, part):
    if len(part.sectionAssignments) == 0:
        raise RuntimeError("No sectionAssignments found on target part '%s'." % part.name)
    sec_name = part.sectionAssignments[0].sectionName
    if sec_name not in mdl.sections:
        raise RuntimeError("Section '%s' not found in mdl.sections." % sec_name)
    base_mat = mdl.sections[sec_name].material
    if not base_mat:
        raise RuntimeError("Could not determine base material from section '%s'." % sec_name)
    return base_mat

def bbox_z_of_part(part):
    bb = part.getBoundingBox()
    zmin = bb["low"][2]
    zmax = bb["high"][2]
    return zmin, zmax

# ============================================================
# 3) DISKS: ensure two disk instances exist and are placed
# ============================================================

def ensure_disks(mdl, specimen_part_name):
    """
    If Disque-1 / Disque-2 already exist in assembly -> do nothing.
    Else create them from part 'Disque' and place them at zmin/zmax of specimen.
    """
    a = mdl.rootAssembly

    if DISK_PART_NAME not in mdl.parts:
        print("WARNING: Disk part '%s' not found. Skipping disk creation." % DISK_PART_NAME)
        return

    # already there?
    if (DISK_INST_BOTTOM in a.instances) and (DISK_INST_TOP in a.instances):
        print("--- Disks already exist in assembly (%s, %s). Keeping them." % (DISK_INST_BOTTOM, DISK_INST_TOP))
        return

    if specimen_part_name not in mdl.parts:
        print("WARNING: Specimen part '%s' not found in mdl.parts. Skipping disk placement." % specimen_part_name)
        return

    specimen = mdl.parts[specimen_part_name]
    disk = mdl.parts[DISK_PART_NAME]

    zmin_spec, zmax_spec = bbox_z_of_part(specimen)
    zmin_disk, zmax_disk = bbox_z_of_part(disk)

    # Create missing instances
    if DISK_INST_BOTTOM not in a.instances:
        a.Instance(name=DISK_INST_BOTTOM, part=disk, dependent=ON)
    if DISK_INST_TOP not in a.instances:
        a.Instance(name=DISK_INST_TOP, part=disk, dependent=ON)

    # Place bottom disk so that its TOP face touches specimen bottom (zmin_spec)
    # disk top face is at zmax_disk in its local placement
    dz_bottom = (zmin_spec - DISK_GAP) - zmax_disk
    a.translate(instanceList=(DISK_INST_BOTTOM,), vector=(0.0, 0.0, dz_bottom))

    # Place top disk so that its BOTTOM face touches specimen top (zmax_spec)
    # disk bottom face is at zmin_disk
    dz_top = (zmax_spec + DISK_GAP) - zmin_disk
    a.translate(instanceList=(DISK_INST_TOP,), vector=(0.0, 0.0, dz_top))

    print("--- Disks created/placed:")
    print("    %s moved dz=%g" % (DISK_INST_BOTTOM, dz_bottom))
    print("    %s moved dz=%g" % (DISK_INST_TOP, dz_top))

# ============================================================
# 4) KEYWORDBLOCK MATERIAL UTILITIES
# ============================================================

def _norm(s):
    return s.strip().lower()

def _material_header_matches(line, matname):
    low = _norm(line).replace(" ", "")
    if not low.startswith("*material"):
        return False
    return ("name=%s" % matname.lower()) in low

def _find_material_block(blocks, matname):
    start = None
    for i, ln in enumerate(blocks):
        if _material_header_matches(ln, matname):
            start = i
            break
    if start is None:
        return None, None
    end = len(blocks)
    for j in range(start + 1, len(blocks)):
        if _norm(blocks[j]).startswith("*material"):
            end = j
            break
    return start, end

def remove_material_block_if_exists(kb, matname):
    try:
        kb.synchVersions(storeNodesAndElements=False)
        blocks = list(kb.sieBlocks)
        start, end = _find_material_block(blocks, matname)
        if start is None:
            return
        for idx in range(start, end):
            kb.replace(idx, "")
        kb.synchVersions(storeNodesAndElements=False)
    except Exception:
        pass

def build_cdp_material_text(md):
    E   = md["E"]
    nu  = md["nu"]
    fc0 = md["fc0"]
    ft0 = md["ft0"]

    f1   = _R_F1 * fc0
    cres = _R_CRES * fc0
    tres = _R_TRES * ft0

    lines = []
    lines.append("*Material, name=%s" % md["name"])

    lines.append("*Elastic")
    lines.append("%g, %g" % (E, nu))

    lines.append("*Concrete Damaged Plasticity")
    lines.append("%g, %g, %g, %g, %g" % (CDP_DILAT, CDP_ECC, CDP_FBFC, CDP_K, CDP_VISC))

    lines.append("*Concrete Compression Hardening")
    lines.append(" %g, %g" % (f1, 0.0))
    lines.append(" %g, %g" % (fc0, _EIN_PIC))
    lines.append(" %g, %g" % (cres, _EIN_END))

    lines.append("*Concrete Tension Stiffening")
    lines.append(" %g, %g" % (ft0, 0.0))
    lines.append(" %g, %g" % (tres, _ETIN_END))

    lines.append("*Concrete Compression Damage")
    for d, eps in DC_POINTS:
        lines.append(" %g, %g" % (d, eps))

    lines.append("*Concrete Tension Damage")
    for d, eps in DT_POINTS:
        lines.append(" %g, %g" % (d, eps))

    return "\n".join(lines)

def kb_insert_robust(kb, text):
    kb.synchVersions(storeNodesAndElements=False)
    n = len(kb.sieBlocks)

    # Try safe insert positions
    candidates = []
    if n > 0:
        candidates += [n - 1, n - 2]
    candidates += [-1, 0]

    last_err = None
    for idx in candidates:
        try:
            kb.insert(idx, text)
            kb.synchVersions(storeNodesAndElements=False)
            return
        except Exception as e:
            last_err = e

    raise RuntimeError("Could not insert into keywordBlock. Last error: %s" % str(last_err))

# ============================================================
# 5) SECTIONS + HETEROGENEITY ASSIGNMENT
# ============================================================

def ensure_sections(mdl):
    for sname in SEC_NAMES:
        safe_del(mdl.sections, sname)
    mdl.HomogeneousSolidSection(name=SEC_NAMES[0], material=minerals[0]["name"])
    mdl.HomogeneousSolidSection(name=SEC_NAMES[1], material=minerals[1]["name"])

def heterogenize_part(part):
    elems = part.elements[:]
    if len(elems) == 0:
        raise RuntimeError("Target part '%s' has no elements." % part.name)

    random.seed(RANDOM_SEED)
    sigma = math.sqrt(2.0) / a

    waves = []
    for _ in range(Nwaves):
        waves.append((random.gauss(0.0, sigma),
                      random.gauss(0.0, sigma),
                      random.gauss(0.0, sigma),
                      2.0 * math.pi * random.random()))
    coef = math.sqrt(2.0 / float(Nwaves))

    def f_point(x, y, z):
        s = 0.0
        for (ox, oy, oz, phi) in waves:
            s += math.cos(ox*x + oy*y + oz*z + phi)
        return coef * s

    vols, fvals = [], []
    V_tot = 0.0
    for e in elems:
        v = element_volume(part, e)
        vols.append(v); V_tot += v

        fv = 0.0
        nn = float(len(e.connectivity))
        for nl in e.connectivity:
            x, y, z = node_xyz(part, nl)
            fv += f_point(x, y, z)
        fvals.append(fv / nn)

    V_targets = [p_frac[0]*V_tot, p_frac[1]*V_tot]

    idx = list(range(len(elems)))
    idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

    elsets = [[], []]
    Vacc = [0.0, 0.0]
    for i in idx:
        if Vacc[0] < V_targets[0]:
            elsets[0].append(elems[i]); Vacc[0] += vols[i]
        else:
            elsets[1].append(elems[i]); Vacc[1] += vols[i]

    for nm in ELSET_NAMES:
        safe_del(part.sets, nm)

    for k in range(2):
        labels = [e.label for e in elsets[k]]
        if len(labels) == 0:
            raise RuntimeError("Mineral-%d empty. Adjust p_frac/a/Nwaves." % (k+1))
        ek = part.elements.sequenceFromLabels(labels=labels)
        part.Set(name=ELSET_NAMES[k], elements=ek)
        part.SectionAssignment(region=regionToolset.Region(elements=ek),
                               sectionName=SEC_NAMES[k])

    fracs = [Vacc[i]/V_tot for i in range(2)]
    print("--- Heterogeneity assigned on part:", part.name)
    print("--- Volume fractions ~", fracs)
    return fracs

# ============================================================
# 6) WRITE OUTPUT INP
# ============================================================

def write_inp(mdl, out_inp_path):
    jname = "JOB_EXPORT_HETERO"
    if jname in mdb.jobs:
        del mdb.jobs[jname]
    job = mdb.Job(name=jname, model=mdl.name, type=ANALYSIS)
    job.writeInput(consistencyChecking=OFF)
    gen = jname + ".inp"

    out_abs = os.path.abspath(out_inp_path)
    gen_abs = os.path.abspath(gen)

    if gen_abs != out_abs:
        try:
            if os.path.exists(out_abs):
                os.remove(out_abs)
        except Exception:
            pass
        os.rename(gen_abs, out_abs)

    print("--- Wrote:", out_abs)

# ============================================================
# 7) ARG PARSING + MAIN
# ============================================================

def parse_args():
    argv = list(sys.argv)
    in_inp = None
    out_inp = None

    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        if len(args) >= 2:
            in_inp = args[0]
            out_inp = args[1]
    else:
        in_inp = DEFAULT_INP
        out_inp = DEFAULT_OUT

    return in_inp, out_inp

def main():
    in_inp, out_inp = parse_args()

    if (not in_inp) or (not out_inp):
        raise RuntimeError(
            "Usage:\n"
            "  abaqus cae noGUI=script.py -- input.inp output.inp\n"
            "Or run in GUI with DEFAULT_INP/DEFAULT_OUT configured."
        )

    if not os.path.exists(in_inp):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, in_inp)
        if os.path.exists(alt):
            in_inp = alt
        else:
            raise RuntimeError("Input .inp not found: %s" % in_inp)

    if not os.path.isabs(out_inp):
        out_inp = os.path.join(os.path.dirname(os.path.abspath(in_inp)), out_inp)

    # Unique model name (avoid GUI KeyError)
    model_name = MODEL_NAME
    k = 1
    while model_name in mdb.models:
        model_name = "%s_%d" % (MODEL_NAME, k)
        k += 1

    print("--- Importing:", in_inp)
    mdl = mdb.ModelFromInputFile(name=model_name, inputFileName=in_inp)
    print("--- Using model name:", model_name)

    part = pick_target_part(mdl)
    print("--- Target specimen part:", part.name, " (#elems=", len(part.elements), ")")

    # Ensure disks exist/placed (if missing)
    ensure_disks(mdl, specimen_part_name=part.name)

    base_mat = get_base_material_from_part(mdl, part)
    print("--- Base material detected:", base_mat)

    kb = mdl.keywordBlock
    kb.synchVersions(storeNodesAndElements=False)

    # Remove existing Mineral-1/2 blocks if present
    for md in minerals:
        remove_material_block_if_exists(kb, md["name"])

    # Insert new CDP materials
    for md in minerals:
        txt = build_cdp_material_text(md)
        kb_insert_robust(kb, txt)

    ensure_sections(mdl)
    heterogenize_part(part)
    write_inp(mdl, out_inp)

    print("--- DONE. Output:", out_inp)

main()