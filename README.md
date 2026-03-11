# PhD Thesis — Optimization of Deep Geothermal Drilling

**Coupled Modeling of Percussion and Shearing Mechanisms for Enhanced Cutting Performance**

> *Fernando Uzquiano Al-Ricabi*  
> Mines Paris – PSL, Centre de Géosciences, Fontainebleau, France  
> Supervisors: [Supervisor names]  
> Start: September 2025 · Expected defense: 2028

[![Institution](https://img.shields.io/badge/Mines%20Paris%20–%20PSL-Centre%20de%20Géosciences-blue?style=flat)](https://www.minesparis.psl.eu)
[![Project](https://img.shields.io/badge/Project-HyperDrill-orange?style=flat)]()
[![Status](https://img.shields.io/badge/Status-Year%201-green?style=flat)]()

---

## Context & Motivation

Deep geothermal energy is one of the few renewable sources capable of delivering **baseload, carbon-free power** independent of weather conditions. Scaling it up globally requires drilling reliably into **hard crystalline rock** (granite, basalt) at depths exceeding 3–5 km — a regime where conventional rotary drilling tools suffer from rapid wear and very low Rate of Penetration (ROP).

**Percussive drilling** — combining high-frequency impact loading with rotary cutting — is a promising alternative that can significantly increase ROP in hard formations. However, the coupled mechanics of percussion and shearing at the bit-rock interface remain poorly understood, limiting systematic optimization.

This thesis addresses that gap through physics-based numerical modeling.

---

## Research Questions

```
1. How do percussion and shearing mechanisms interact at the bit-rock interface?
2. Can a coupled FEM model capture this interaction with sufficient predictive accuracy?
3. What combination of drilling parameters (impact energy, WOB, RPM, bit geometry)
   maximizes ROP while minimizing specific energy and tool wear?
```

---

## Methodology

### Coupled Percussion–Shear FEM Model

The core of this thesis is the development of a **finite element model** capturing the simultaneous effect of:

- **Percussive loading** — dynamic impact forces, stress wave propagation in the rock, crater formation
- **Rotary cutting** — shear-induced chip generation, frictional contact at the bit-rock interface
- **Coupling effects** — how impact weakens the rock ahead of the cutting edge, and how rotation affects stress redistribution between impacts

**Rock constitutive behavior** is modeled using elastoplastic frameworks adapted to brittle hard rock, accounting for confinement pressure effects relevant to deep drilling conditions.

### Planned Workflow

```
Literature review & model design
        ↓
Benchmark: reproduce published single-indenter results
        ↓
Coupled percussion–shear 2D model (axisymmetric)
        ↓
Extension to 3D bit geometry
        ↓
Parametric study → optimization
        ↓
Experimental validation (lab + HyperDrill prototype)
```

---

## HyperDrill Collaboration

This PhD is embedded in the **HyperDrill** international research project:

| Partner | Contribution |
|---|---|
| **Mines Paris – PSL** | Numerical modeling, constitutive modeling, optimization |
| **TU Clausthal** (Germany) | Experimental platform, prototype drilling rig, field-scale validation |

The Franco-German structure allows tight coupling between model development and experimental feedback — a key methodological asset of this project.

---

## Tools & Environment

| Category | Tools |
|---|---|
| FEM | PLAXIS, CLEO 2D-CESAR |
| Scripting & automation | Python (NumPy, SciPy, Matplotlib) |
| Optimization | Gradient-free methods, parametric sweeps |
| Document preparation | LaTeX |

---

## Repository Structure *(in progress)*

```
📦 phd-hyperdrill/
 ┣ 📂 literature/      # Annotated references and state-of-the-art notes
 ┣ 📂 numerical/       # FEM models and simulation scripts
 ┣ 📂 figures/         # Schematics and plots
 ┣ 📂 reports/         # Progress reports, meeting notes (HyperDrill)
 └ 📜 README.md
```

> This repository is at an early stage. Code and models will be progressively added as the research develops.

---

## Publications

| Reference | Status |
|---|---|
| *Coupled percussion–shear FEM modeling for deep geothermal drilling optimization*, F. Uzquiano Al-Ricabi et al. | In preparation |

---

## Contact

Open to discussion on bit-rock interaction modeling, hard rock mechanics, or geothermal drilling — feel free to reach out.

- 📧 [fernandouzquiano@outlook.fr](mailto:fernandouzquiano@outlook.fr)
- 💼 [linkedin.com/in/fernando-uzquiano](https://www.linkedin.com/in/fernando-uzquiano)
- 🏛️ Mines Paris – PSL, Centre de Géosciences — Fontainebleau, France

---

*Code released under MIT License · Academic content © Fernando Uzquiano Al-Ricabi*
