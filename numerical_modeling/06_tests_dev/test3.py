# -*- coding: utf-8 -*-
# make_hetero_from_inp.py
#
# Import an existing homogeneous .inp, keep EVERYTHING, but turn the main cylinder part into
# a 3-mineral heterogeneous solid using cosine-wave Gaussian field (Cahn/Lantuéjoul style),
# with element value = nodal average, and thresholding to match target fractions (volume-weighted).
#
# Run:
#   abaqus cae noGUI=make_hetero_from_inp.py -- input.inp output.inp
#
# Notes:
# - The script tries to auto-detect the cylinder part (largest 3D deformable part by #elements).
# - If you want to force a part name, set FORCE_PART_NAME below.

from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
import random
import math
import os
import sys

# ============================================================
# 1) USER PARAMETERS
# ============================================================

MODEL_NAME = 'Model-1'

# If not None, force the part name (e.g. 'Part-1' or 'CYL')
FORCE_PART_NAME = None  # ex: 'Part-1'

# Random field
RANDOM_SEED = 42
a = 3.0
Nwaves = 300
MIN1_IS_LOW = True

# Target fractions (sum=1)
p_frac = (0.30, 0.50, 0.20)

# Materials (E in MPa, rho in t/mm^3 if you follow mm-MPa-tonne)
materials = [
    dict(name='Mineral-1', E=35000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-2', E=72000.0, nu=0.25, rho=2.62e-9),
    dict(name='Mineral-3', E=52000.0, nu=0.25, rho=2.62e-9),
]

# Naming
SEC_NAMES = ['SEC_MIN1', 'SEC_MIN2', 'SEC_MIN3']
ELSET_NAMES = ['ELSET_MIN1', 'ELSET_MIN2', 'ELSET_MIN3']


# ============================================================
# 2) HELPERS
# ============================================================

def element_volume(part, elem):
    """Prefer Abaqus getSize(); fallback to bbox volume."""
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
    """Pick the 'main' deformable 3D part: by default the one with the largest element count."""
    if FORCE_PART_NAME is not None:
        if FORCE_PART_NAME not in mdl.parts:
            raise RuntimeError("FORCE_PART_NAME='%s' not found in model.parts." % FORCE_PART_NAME)
        return mdl.parts[FORCE_PART_NAME]

    best = None
    best_ne = -1
    for pname, p in mdl.parts.items():
        # Only deformable 3D solids make sense here
        try:
            if p.dimensionality != THREE_D:
                continue
            if p.type != DEFORMABLE_BODY:
                continue
        except Exception:
            # Some Abaqus objects may not expose these safely; ignore
            pass

        ne = len(p.elements)
        if ne > best_ne:
            best_ne = ne
            best = p

    if best is None or best_ne <= 0:
        raise RuntimeError("Could not auto-detect a 3D deformable meshed part.")
    return best

def safe_delete(container, key):
    try:
        if key in container:
            del container[key]
    except Exception:
        pass

def ensure_materials_and_sections(mdl):
    # Create / replace materials
    for md in materials:
        safe_delete(mdl.materials, md['name'])
    for i in range(3):
        safe_delete(mdl.sections, SEC_NAMES[i])

    for i, md in enumerate(materials):
        mat = mdl.Material(name=md['name'])
        mat.Density(table=((md['rho'],),))
        mat.Elastic(table=((md['E'], md['nu']),))
        mdl.HomogeneousSolidSection(name=SEC_NAMES[i], material=md['name'])

def build_random_waves(seed, a_mm, nwaves):
    random.seed(seed)
    sigma = math.sqrt(2.0) / a_mm
    waves = []
    for _ in range(nwaves):
        waves.append((random.gauss(0.0, sigma),
                      random.gauss(0.0, sigma),
                      random.gauss(0.0, sigma),
                      2.0 * math.pi * random.random()))
    return waves

def heterogenize_part_3minerals(part):
    elems = part.elements[:]
    if len(elems) == 0:
        raise RuntimeError("Target part has no elements. Mesh must exist in the imported .inp.")

    # Build Gaussian-like field via cosine waves
    waves = build_random_waves(RANDOM_SEED, a, Nwaves)
    coef = math.sqrt(2.0 / float(Nwaves))

    def f_point(x, y, z):
        s = 0.0
        for (ox, oy, oz, phi) in waves:
            s += math.cos(ox*x + oy*y + oz*z + phi)
        return coef * s

    # Compute element volumes + element field value (nodal average)
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

    # Targets (volume-weighted)
    V_targets = [p_frac[0]*V_tot, p_frac[1]*V_tot, p_frac[2]*V_tot]

    # Sort indices by field
    idx = list(range(len(elems)))
    idx.sort(key=lambda i: fvals[i], reverse=(not MIN1_IS_LOW))

    # Fill element groups until targets reached
    elsets = [[], [], []]
    Vacc = [0.0, 0.0, 0.0]
    for i in idx:
        if Vacc[0] < V_targets[0]:
            elsets[0].append(elems[i]); Vacc[0] += vols[i]
        elif Vacc[1] < V_targets[1]:
            elsets[1].append(elems[i]); Vacc[1] += vols[i]
        else:
            elsets[2].append(elems[i]); Vacc[2] += vols[i]

    # Create/replace Elsets and section assignment
    for nm in ELSET_NAMES:
        safe_delete(part.sets, nm)

    for k in range(3):
        labels = [e.label for e in elsets[k]]
        if len(labels) == 0:
            raise RuntimeError("Mineral-%d empty. Adjust p_frac / a / Nwaves." % (k+1))

        ek = part.elements.sequenceFromLabels(labels=labels)
        part.Set(name=ELSET_NAMES[k], elements=ek)
        part.SectionAssignment(region=regionToolset.Region(elements=ek),
                               sectionName=SEC_NAMES[k])

    frac = [Vacc[i]/V_tot for i in range(3)]
    print('--- Heterogeneity assigned on part:', part.name)
    print('--- V_tot =', V_tot)
    print('--- Volume fractions ~', frac)
    return frac


def write_out_inp(mdl, out_inp_path):
    # Create a job and write input
    jname = 'JOB_HETERO_EXPORT'
    if jname in mdb.jobs:
        del mdb.jobs[jname]
    job = mdb.Job(name=jname, model=mdl.name, type=ANALYSIS)
    # This writes <jobname>.inp in cwd. We'll rename/move to out_inp_path.
    job.writeInput(consistencyChecking=OFF)
    generated = jname + '.inp'

    # Move/rename
    if os.path.abspath(generated) != os.path.abspath(out_inp_path):
        try:
            if os.path.exists(out_inp_path):
                os.remove(out_inp_path)
        except Exception:
            pass
        os.rename(generated, out_inp_path)

    print('--- Wrote heterogeneous inp to:', out_inp_path)


# ============================================================
# 3) MAIN
# ============================================================

def main():
    # Parse args after "--"
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--')+1:]
    else:
        argv = argv[1:]

    if len(argv) != 2:
        raise RuntimeError("Usage: abaqus cae noGUI=make_hetero_from_inp.py -- input.inp output.inp")

    in_inp = argv[0]
    out_inp = argv[1]

    if not os.path.exists(in_inp):
        raise RuntimeError("Input .inp not found: %s" % in_inp)

    # Clean existing model if any
    if MODEL_NAME in mdb.models:
        # safer to delete and reimport
        del mdb.models[MODEL_NAME]

    # Import INP (keeps steps/BC/loads/contact as defined in the file)
    print('--- Importing inp:', in_inp)
    mdl = mdb.ModelFromInputFile(name=MODEL_NAME, inputFileName=in_inp)

    # Ensure materials+sections exist (and are named as we want)
    ensure_materials_and_sections(mdl)

    # Pick part (auto or forced)
    p = pick_target_part(mdl)
    print('--- Target part selected:', p.name, ' (#elems=', len(p.elements), ')')

    # Apply heterogeneity + section assignments
    heterogenize_part_3minerals(p)

    # Export new inp
    write_out_inp(mdl, out_inp)

    print('--- DONE.')

main()