# Physics AI for Oil & Gas: A State-of-the-Field Reference

*A living reference document surveying physics-informed and scientific machine learning (SciML) as applied to subsurface reservoir modeling, seismic imaging, drilling, production, pipeline flow, and refining. Compiled September 2026 as a seed document for an open-source community reference.*

---

## Table of Contents

1. [Introduction & Scope](#introduction--scope)
2. [Academic Research Landscape](#academic-research-landscape)
3. [Model Architectures](#model-architectures)
4. [Commercial Products & Startups](#commercial-products--startups)
5. [Numerical Methods & Solvers (Baselines and Hybrids)](#numerical-methods--solvers-baselines-and-hybrids)
6. [Geometry, Mesh & Subsurface Model Resources](#geometry-mesh--subsurface-model-resources)
7. [Datasets & Data Generators](#datasets--data-generators)
8. [Industry Standards & Validation Context](#industry-standards--validation-context)
9. [Open Challenges & Gaps](#open-challenges--gaps)
10. [Roadmap: Building the Reference Hub](#roadmap-building-the-reference-hub)
11. [Sources](#sources)

---

## Introduction & Scope

"Physics AI" in this document means machine learning methods that explicitly incorporate physical laws — partial differential equations (PDEs), conservation laws, constitutive relations — into model architecture, training loss, or data generation, as opposed to purely data-driven black-box ML. In the oil & gas (O&G) context this spans five broad application domains:

- **Subsurface reservoir modeling**: multiphase Darcy flow, history matching, production forecasting, CO2 storage/CCUS.
- **Seismic imaging**: full waveform inversion (FWI), velocity model building, fault/horizon interpretation.
- **Drilling**: geosteering, real-time drilling optimization, stuck-pipe/kick detection.
- **Production & facilities**: multiphase pipeline flow, artificial lift, digital twins.
- **Refining/downstream**: reactor kinetics, process optimization (lighter coverage here; the document is upstream-weighted since that is where most physics-informed SciML activity concentrates).

The two dominant technical paradigms recurring across all domains are **Physics-Informed Neural Networks (PINNs)**, which embed PDE residuals into the loss function of a pointwise neural network via automatic differentiation, and **neural operators** (Fourier Neural Operator/FNO, DeepONet, and variants), which learn mappings between function spaces (e.g., permeability field → pressure field) and can generalize across geometries/parameters after training, unlike PINNs which are typically solved per-instance.

---

## Academic Research Landscape

### Foundational surveys
The clearest entry points into this literature are two survey papers:
- **"Applications of physics-informed scientific machine learning in subsurface science: A survey"** (arXiv:2104.04764) — an early, broad framing of PIML for subsurface energy systems including fossil fuel exploration, CO2 sequestration, and geothermal.
- **"A Critical Review of Physics-Informed Machine Learning Applications in Subsurface Energy Systems"** (arXiv:2308.04457) — more recent and more critical, cataloguing where PINNs succeed and where they still underperform classical solvers.
- **"Emerging applications of physics-informed and physics-guided machine learning in geoenergy science: A review"** (ScienceDirect, S1007570425009608) covers geothermal, CO2 storage, and hydrocarbon recovery together, reflecting how the same PIML toolkit is now shared across the energy transition and traditional O&G.
- **GMD, "Perspectives of physics-based machine learning strategies for geoscientific applications governed by partial differential equations"** (gmd.copernicus.org/articles/16/7375/2023) situates subsurface flow within the broader geoscience PDE-ML landscape (also touching atmospheric and climate science methods that migrate into reservoir engineering).

### Reservoir simulation / Darcy & multiphase flow
- **PINN-based reservoir simulation with sparse data using domain decomposition** (ScienceDirect S1995822623002947) — addresses a core PINN weakness (poor scaling to large heterogeneous domains) via domain decomposition (PINN-DD), a recurring theme since vanilla PINNs struggle to resolve sharp saturation fronts and permeability contrasts.
- **WellPINN: Accurate Well Representation for Transient Fluid Pressure Diffusion in Subsurface Reservoirs with PINNs** (arXiv:2507.09330) — tackles the well-singularity problem: point-source/sink wells create pressure singularities that generic PINNs represent poorly, a practically important limitation for any operator wanting to use PINNs for real drainage-radius calculations.
- **"A novel reservoir simulation model based on physics informed neural networks"**, *Physics of Fluids* (AIP, 2024) — demonstrates PINNs solving Darcy-law-constrained seepage equations directly.
- **Physics-informed graph neural network for predicting fluid flow in porous media** (ScienceDirect S199582262500216X) and the companion **"Physics-informed graph neural network for spatial-temporal production forecasting"** (arXiv:2209.11885) — combine GNN well-network representations with physical constraints for production forecasting, discussed further under Model Architectures.
- **Transfer learning-based physics-informed CNN for flow in porous media with time-varying controls** (arXiv:2310.06319) — targets the well-control/time-varying-boundary-condition problem that makes reservoir simulation harder than a static PDE-solving benchmark.

### Neural operators for subsurface flow
- **"Fourier Neural Operator for Solving Subsurface Oil/Water Two-Phase Flow Partial Differential Equation"**, *SPE Journal* (OnePetro, 2022) — one of the first FNO papers to appear in a core petroleum-engineering venue rather than a pure ML venue, signaling the field's uptake by SPE.
- **U-FNO (U-shaped Fourier Neural Operator)** for three-phase flow reservoir simulation (ResearchGate 369854804) — reports up to ~3600x speedup versus traditional numerical solvers with minimal accuracy loss, on CO2-water-gas systems relevant to both EOR and CCUS.
- **"Multi-fidelity Fourier neural operator for fast modeling of large-scale geological carbon storage"** (ScienceDirect S0022169424000350) — addresses FNO's grid-resolution sensitivity via multi-fidelity training.
- **"Enhancing subsurface multiphase flow simulation with Fourier neural operator"** (ScienceDirect S2405844024141345 / PubMed 39381234) — reports three-orders-of-magnitude speedups after moderate GPU training versus traditional PDE solvers.
- **Stanford-linked NVIDIA/TotalEnergies collaboration**: "A Novel A.I Enhanced Reservoir Characterization with a Combined Mixture of Experts — NVIDIA Modulus based Physics Informed Neural Operator Forward Model" (arXiv:2404.14447) and its follow-up "Reservoir History Matching of the Norne field with generative exotic priors and a coupled Mixture of Experts — Physics Informed Neural Operator Forward Model" (arXiv:2406.00889) — apply PINO (Physics-Informed Neural Operator) forward models to a real, named field (Norne, offshore Norway) coupled to ensemble history-matching, one of the more complete "real field" demonstrations in the open literature.
- **"A reduced-order derivative-informed neural operator for subsurface fluid-flow"** (arXiv:2509.13620) — recent (2025) work on gradient-aware operator learning intended to make neural-operator surrogates usable inside gradient-based history matching / optimization loops.

### Seismic imaging & full waveform inversion (FWI)
- **OpenFWI** (arXiv:2111.02926, and its elastic extension OpenFWI 2.0 / "E-FWI", arXiv:2306.12386) — the standard large-scale (12 datasets, ~2.1 TB) benchmark suite for deep-learning FWI, from a Microsoft Research/academic collaboration, enabling reproducible comparison across supervised, unsupervised, and physics-driven methods.
- **DeepONet / operator-learning for FWI**: "Leveraging Deep Operator Networks (DeepONet) for Acoustic FWI"; **Fourier-DeepONet** (ScienceDirect S0045782523004243), which improves generalizability/robustness of DeepONet for FWI; **En-DeepONet** (ScienceDirect S0045782523008046), an enriched architecture for moving-source operators applied to the Eikonal equation and instant traveltime prediction; **SG-DeepONet** (arXiv:2408.08005) for source-generalized FWI.
- **Deep Neural Helmholtz Operators for 3D Elastic Wave Propagation and Inversion** (arXiv:2311.09608, *Geophysical Journal International*) — extends neural-operator wave modeling to full 3D elastic media, closer to field-scale seismic problems than 2D acoustic toy cases.
- **Physics-guided/hybrid FWI**: "Physics-guided Full Waveform Inversion using Encoder-Solver CNNs" (arXiv:2405.17696), "Seismic full waveform inversion via a physics-guided Fourier representation neural network," and the survey "Synergizing Deep Learning and Full-Waveform Inversion: Bridging Data-Driven and Theory-Guided Approaches for Enhanced Seismic Imaging" (arXiv:2502.17585) — this last one is a useful 2025 synthesis of where hybrid (physics + data) FWI stands relative to pure classical FWI and pure deep-learning FWI.
- **Uncertainty-aware FWI**: "Conditional Image Prior for Uncertainty Quantification in Full Waveform Inversion" (arXiv:2408.09975) and "Accelerating seismic inversion and uncertainty quantification with efficient high-rank Hessian approximations" (arXiv:2507.10804) — UQ for FWI is an active subfield because velocity-model ambiguity directly affects prospect risking.
- **Devito** (devitoproject.org, from Imperial College London) is the standard open-source finite-difference DSL used across much of this literature to generate FDTD reference/training data for wave-equation ML research; several of the DeepONet/FNO wave papers cite it as their forward-solver backbone.
- **Survey**: "Survey of Deep Learning and Physics-Based Approaches in Computational Wave Imaging" (arXiv:2410.08329) is a good broad map of the FWI-ML space as of late 2024.

### Key venues
- **SPE (Society of Petroleum Engineers)**: *SPE Journal*, *SPE Reservoir Evaluation & Engineering*, *SPE Drilling & Completion* (all on OnePetro), plus flagship conferences **SPE ATCE** (Annual Technical Conference and Exhibition), **SPE Reservoir Simulation Conference**, and **IPTC** (International Petroleum Technology Conference) — increasingly carry PINN/FNO papers alongside traditional simulation papers (e.g., the SPE Journal FNO two-phase-flow paper above, and IPTC 2026 "AI Accelerated Forward Modeling Using Physics Informed Neural Operators for Real-Time Reservoir Management").
- **Geophysics** (SEG's flagship journal) and **Geophysical Journal International** (GJI, RAS/EAGE-linked) publish the bulk of ML-FWI methods papers.
- **SEG** (Society of Exploration Geophysicists): its **Advancing Data Analytics & Machine Learning for Exploration Geophysics** workshop (next edition May 18–21, 2026, Houston) is a recurring dedicated venue.
- **EAGE** (European Association of Geoscientists & Engineers): its **Workshop on Enhancing Subsurface Practices using AI/ML** (e.g., held Nov 10–11, 2025, Perth) and general EAGE Annual Conference machine-learning tracks; EarthDoc (earthdoc.org) is EAGE's paper repository.
- **arXiv** categories **physics.geo-ph** and **cs.LG** now host most cutting-edge PIML-for-geoscience preprints months before journal publication — a practical reality any reference repo should account for by tracking arXiv alongside peer-reviewed venues.

---

## Model Architectures

| Architecture | Governing physics / use case | Representative work |
|---|---|---|
| **PINN (Physics-Informed Neural Network)** | Darcy flow, Buckley-Leverett two-phase displacement, well-pressure diffusion | WellPINN (arXiv:2507.09330); PINN-DD domain decomposition (ScienceDirect S1995822623002947) |
| **PINN for poroelasticity** | Coupled flow-geomechanics (Biot's equations), fixed-stress-split sequential training | "Physics-informed neural network simulation of multiphase poroelasticity using stress-split sequential training" (arXiv:2110.03049); "Physics-informed deep learning for flow and deformation in poroelastic media" (arXiv:2010.15426) |
| **Fourier Neural Operator (FNO) / U-FNO** | Two- and three-phase Darcy flow, CO2-water-gas systems, CCUS | SPE Journal FNO paper (OnePetro 480880); U-FNO (ResearchGate 369854804); multi-fidelity FNO for carbon storage (ScienceDirect S0022169424000350) |
| **DeepONet family (DeepONet, Fourier-DeepONet, En-DeepONet, SG-DeepONet)** | Operator learning for FWI, Eikonal traveltimes, moving-source wave problems | ScienceDirect S0045782523004243; S0045782523008046; arXiv:2408.08005 |
| **Physics-Informed Neural Operator (PINO)** | Hybrid: operator-learning speed + PDE-residual regularization for reservoir forward modeling | NVIDIA Modulus-based Norne field history matching (arXiv:2404.14447, arXiv:2406.00889) |
| **CNN/U-Net for seismic** | Impedance/velocity inversion, image-to-image FWI surrogates | "Improving the Seismic Impedance Inversion by Fully Convolutional Neural Network" (MDPI jmse13020262); encoder-solver CNN FWI (arXiv:2405.17696) |
| **Graph Neural Networks (GNN/GCN)** | Well-to-well interference, waterflood inter-well connectivity, field-level production forecasting | Physics-informed GNN for spatial-temporal production forecasting (arXiv:2209.11885); GCN-RNN hybrid for waterflooded reservoirs (SPE IOR 26IOR/D011S002R001); "Dynamic interwell connectivity analysis... improved graph neural network" (ResearchGate 375585792) |
| **Transformer / time-series foundation models** | Well-log imputation, formation-top interpretation, anomaly detection, cross-well/cross-basin generalization | WLFM: A Well-Logs Foundation Model (arXiv:2509.18152) — VQ-tokenized "geological vocabulary"; TGS "Well log foundation model: Making promptable AI models for interpretation"; "Leveraging Time-Series Foundation Model for Subsurface Well Logs Prediction" (arXiv:2412.05681, fine-tuning TimeGPT); a 60M-parameter masked-autoencoder well-log vision transformer pretrained on 1.1 million North American well logs |
| **Generative models (GAN, diffusion/rectified flow)** | Geological realization generation, seismic-to-velocity domain transfer, FWI as a generative inverse problem | "Rapid seismic domain transfer... using deep generative neural networks" (arXiv:1805.08826); "Seismic full-waveform inversion based on a physics-driven GAN" (arXiv:2603.14879); "Conditional Rectified Flow-based End-to-End Rapid Seismic Inversion" (arXiv:2603.15354); continuous conditional GANs for poroelasticity (arXiv:2111.14984) |
| **Coupled flow-geomechanics surrogates** | CO2 sequestration deformation-flow coupling | "Deep-learning-based coupled flow-geomechanics surrogate model for CO2 sequestration" (arXiv:2105.01334) |

A useful framing that recurs across the review papers: **PINNs** solve one instance of a PDE at a time (good for data-sparse single-well or single-field problems, poor for repeated what-if scenario runs), while **neural operators** (FNO, DeepONet, PINO) amortize training cost across an entire family of PDE instances/geological realizations, which is the shape reservoir engineers actually need for history matching, optimization under uncertainty, and real-time decision support — this is why FNO/PINO work is scaling faster toward field deployment (e.g., TotalEnergies/NVIDIA Norne field work) than pointwise PINNs.

---

## Commercial Products & Startups

### Majors' internal / vendor-integrated AI
- **SLB (formerly Schlumberger)** — **DELFI** cognitive E&P environment packages ML across the petrotechnical workflow; **Petrel** now ships on-prem ML/AI capabilities (fault interpretation, horizon prediction, facies/property modeling) previously cloud-only on DELFI; **INTERSECT** is SLB's high-resolution reservoir simulator, increasingly paired with AI-enabled workflows for faster compute and decision support. SLB also owns **OLGA**, the industry-standard dynamic multiphase-flow pipeline simulator, and **PIPESIM** for steady-state network/wellbore modeling.
- **Halliburton / Landmark** — **DecisionSpace 365** (subsurface modeling) and **LOGIX** (geosteering) are Halliburton's flagship digital/AI-enabled product lines; Landmark's DecisionSpace Platform targets large-scale collaboration across legacy and real-time upstream data.
- **Emerson** — acquired **Paradigm** (2017, ~$510M) and previously owned **Roxar**, combining them into an end-to-end E&P software portfolio (seismic processing/interpretation through production modeling). In 2025 Emerson announced **Project Beyond**, a software-defined OT-ready platform for deploying industrial AI applications across automation environments — signaling a push from point-tools toward a platform play similar to SLB's DELFI.
- **Baker Hughes** and **Saudi Aramco** are frequently cited (alongside ADNOC, Shell, ConocoPhillips, Chevron) as upstream organizations with public AI programs, though granular technical detail on their internal physics-ML stacks is less publicly documented than SLB/Halliburton/Emerson's productized tools.
- **NVIDIA** — provides the compute/framework layer rather than a domain product. **NVIDIA Modulus** (recently rebranded/evolved as **PhysicsNeMo**) is the physics-ML framework underlying much of the PINO/FNO reservoir work cited above. NVIDIA's own materials describe PINN-based reactor microkinetics modeling for a global O&G company (up to ~10⁶x GPU speedup claims over CPU-ported solutions in some cases) and reservoir-modeling speedups reported in the 100x–6000x range versus conventional methods. **TotalEnergies** is publicly named as using sequential Physics-Informed Neural Operators built with Modulus, combined with GANs and optimizers, for subsurface modeling including CCUS.

### Seismic-AI specialists
- **Bluware** — cloud-native subsurface data platform with **InteractivAI**, an interactive deep-learning tool where geoscientists label a feature (fault, salt body, channel, injectite, bright spot, shallow hazard) and the network propagates the interpretation across the full seismic volume in real time, starting from an untrained network per project rather than a pre-trained black box. Bluware has partnered with Microsoft Azure on an OSDU-enabled AI seismic interpretation solution for an energy "super major."

### Industrial DataOps / digital twins
- **Cognite** — **Cognite Data Fusion** is an Industrial DataOps platform providing the contextualized data backbone (knowledge graph) for digital twins; used across maintenance/reliability, production optimization, and digital-field-workforce use cases in O&G (example: Cosmo Oil's refinery digital twin consolidating three facilities into one virtual space).
- **Aize** — digital-twin software provider spun out of Aker/Cognite-adjacent Norwegian tech ecosystem; deployed on FPSOs (e.g., ExxonMobil Guyana / SBM Offshore's Liza Unity and Liza Destiny vessels, Stabroek Block) and on drilling rigs; named as one of **bp's** global digital-twin providers; acquired reality-capture/AI specialist **Samp** to bridge digital twins and field asset data.
- **Beyond Limits** — "cognitive AI" vendor combining ML with encoded human/expert knowledge (a hybrid symbolic + ML approach) for oil & gas decision support, including refining-specific cognitive AI products.

### Production analytics / unconventional-focused
- **Novi Labs** — AI-driven predictive analytics for unconventional (shale) well planning: well placement, spacing, completion design, and production forecasting. In URTeC studies with **Equinor**, Novi Labs reported generating forecasts for hundreds of wells in minutes with >30% improved predictive accuracy versus Arps decline-curve baselines. Raised $35M from Invictus Growth Partners; launched "Novi Intelligence," a research platform for energy-investment-grade forecasting transparency.

### Note on unverified/ambiguous names from the original brief
Targeted searches did not surface substantive, verifiable public information specifically tying **Ruths.ai**, **Ceres Imaging**, **Katalyst Data Management**, **Diamond Offshore digital twins**, or **Koloma** to physics-informed AI for oil & gas as described in the brief (Ceres Imaging in particular appears to be an agriculture-imagery company, not O&G; Koloma is a geologic-hydrogen company, adjacent to but distinct from conventional O&G). Rather than fabricate specifics, these are flagged here as **open research items** for the repo community to verify and fill in with primary sources.

---

## Numerical Methods & Solvers (Baselines and Hybrids)

Physics-AI methods are trained against, validated against, and increasingly hybridized with classical numerical solvers. Any serious reference repo needs to catalog these as the "ground truth" layer:

**Reservoir simulation (finite volume / finite difference, fully implicit):**
- **Eclipse** (SLB) — the long-standing commercial industry-standard black-oil/compositional simulator.
- **CMG** (Computer Modelling Group) — IMEX, GEM, STARS simulators for black-oil, compositional, and thermal/EOR processes respectively.
- **INTERSECT** (SLB) — high-resolution, massively parallel next-generation simulator, positioned as Eclipse's successor for giant/complex models.
- **OPM Flow** (Open Porous Media initiative) — the major **open-source** alternative: a fully-implicit three-phase black-oil simulator supporting CO2 storage, thermal simulation, polymer/solvent extensions. Originally founded ~2009 as a collaboration between Equinor (then Statoil), SINTEF, University of Stuttgart, and University of Bergen. Described in "The Open Porous Media Flow Reservoir Simulator" (arXiv:1910.06059) and a companion Computers & Mathematics with Applications paper. Because it is open-source, OPM Flow is the natural baseline solver for a public physics-AI-for-O&G repo — it can generate training/validation data without licensing issues, unlike Eclipse/CMG/INTERSECT.
- **MRST** (MATLAB Reservoir Simulation Toolbox, SINTEF) — open-source research toolbox widely used for upscaling, grid, and reservoir-workflow prototyping (includes direct SPE10 dataset loaders).

**Seismic wave propagation:**
- **Finite-difference time-domain (FDTD)** — the workhorse method for generating synthetic shot-gather training data.
- **Devito** (Imperial College London) — an open-source, symbolic finite-difference DSL widely used as the reference forward-solver in ML-FWI papers (used to generate training/validation wavefields for several DeepONet/FNO wave-equation papers cited above).
- **Spectral-element methods (SEM)** — used for higher-accuracy elastic/anisotropic wave propagation, more expensive than FDTD, sometimes used to generate "gold standard" validation data for neural-operator wave surrogates.

**Geomechanics coupling:**
- **Biot poroelasticity** equations (fixed-stress or iteratively-coupled schemes) remain the standard formulation being learned/approximated by PINN poroelasticity work (arXiv:2110.03049, arXiv:2010.15426) and coupled flow-geomechanics surrogates for CO2 sequestration (arXiv:2105.01334).

**Pipeline / multiphase flow:**
- **OLGA** (SLB) — the de facto industry-standard dynamic multiphase-flow pipeline simulator ("oil and gas simulator"), used for transient flow-assurance modeling across the full field life cycle; frequently appears as the reference/ground-truth model that ML surrogates (MLP-based dynamic surrogates, Bayesian-neural-network-adjusted surrogates) are trained to approximate for faster what-if pipeline-network analysis.
- **PIPESIM** (SLB) — steady-state wellbore/network simulator, often used alongside OLGA.

**Bridging classical and AI:** across nearly every subdomain — reservoir flow, FWI, poroelasticity, pipeline flow — the dominant near-term pattern is not "replace the solver" but **hybridization**: classical solvers (OPM Flow, Devito, OLGA, Eclipse) generate training/validation data and serve as the accuracy reference, while PINN/FNO/DeepONet surrogates provide fast approximate inference for tasks requiring thousands of forward-model evaluations (history matching ensembles, uncertainty quantification, real-time optimization) where re-running the full solver is computationally prohibitive.

---

## Geometry, Mesh & Subsurface Model Resources

- **Corner-point grids** (the de facto standard grid geometry for commercial reservoir simulators like Eclipse, INTERSECT, CMG, and OPM Flow) — represented in industry-standard **GRDECL**-style ASCII formats; MRST and OPM both provide readers/writers.
- **SPE Comparative Solution Project benchmark models** — a series of community-standard test reservoirs maintained via SPE:
  - **SPE9** (9th Comparative Solution Project) — 9,000-cell (24×25×15) rectilinear black-oil benchmark, layered porosity, highly heterogeneous permeability, 25 rate-controlled producers + 1 corner injector; used to "re-examine black-oil simulation."
  - **SPE10** (10th Comparative Solution Project, "Model 2") — the most widely used upscaling/heterogeneity benchmark: 220×60×85 grid (1,122,000 cells), top 35 layers representing the smooth Tarbert formation, bottom 50 layers representing the highly channelized Upper-Ness formation. Freely distributable and packaged as a loadable case in MRST (SINTEF's `spe10` module) and in the OPM test suite (github.com/OPM/opm-tests).
- **Seismic velocity-model benchmarks:**
  - **Marmousi / Marmousi-2** — the canonical synthetic 2D velocity-model benchmark in exploration geophysics, featuring steeply dipping faults, high-velocity salt domes, thin interbedded layers; Marmousi-2 spans 17 km laterally × 3.5 km depth, 199 geological layers, velocities from 1500–4700 m/s, published in *The Leading Edge* ("The Marmousi experience," library.seg.org/doi/10.1190/1.1437051).
  - **SEAM** (SEG Advanced Modeling program) models — more complex 3D synthetic earth models (including salt bodies) used for FWI and imaging-algorithm benchmarking industry-wide.
  - **OpenFWI** (see Academic section) packages multiple synthetic velocity-model families (flat, curved, fault, CO2-reservoir geometries) specifically formatted for deep-learning FWI benchmarking, at 2K–67K samples per dataset and 2.1 TB total.
- **Well log format**: **LAS (Log ASCII Standard)**, maintained by the **Canadian Well Logging Society (CWLS)** — the near-universal digital well-log distribution format. LAS 2.0 is most common in practice; LAS 3.0 extends it to include tops, core data, deviation surveys, and drill-test data. The Python library **lasio** (pypi.org/project/lasio) is the standard open-source reader/writer and a natural dependency for any repo's well-log tooling.
- **OSDU (Open Subsurface Data Universe)** — an industry-standard, cloud-native data platform/schema effort (referenced via the Bluware/Microsoft Azure OSDU-enabled seismic interpretation work above) aiming to standardize subsurface data interoperability across vendors; worth tracking as the emerging data-plumbing standard that physics-AI tools will increasingly need to read/write against.

---

## Datasets & Data Generators

- **Volve field dataset (Equinor)** — the single most important open, real-world field dataset in this space. Discovered 1993, produced 2008–2016 in the Norwegian North Sea (Hugin Formation sandstone reservoir, 2750–3120 m depth), the field was decommissioned and Equinor released ~40,000 files (well logs, seismic surveys, production/injection histories, reservoir static and simulation models, reports) under an **Equinor Open Data Licence** for research, training, and development use. Accessible via Equinor's data-sharing page (equinor.com/energy/volve-data-sharing); community tooling exists for loading it into **ResInsight** (the open-source reservoir visualization tool from Equinor/Ceetron) and other packages. It is the closest thing the field has to a "real field" equivalent of ImageNet, and is heavily used in recent drilling-intelligence LLM/agent research too (e.g., arXiv:2605.00060, "TADI: Tool-Augmented Drilling Intelligence," and arXiv:2503.07440 on stuck-pipe detection).
- **SPE Comparative Solution Project datasets (SPE1, SPE9, SPE10, and others)** — synthetic but community-standard, freely distributable reservoir models (see Geometry section above) used as reproducible ML training/testing grounds.
- **OPM datasets** — the Open Porous Media initiative distributes open reservoir-simulation datasets and test cases (github.com/OPM/opm-tests) alongside its open-source Flow simulator, explicitly intended to support reproducible research.
- **OpenFWI / OpenFWI 2.0 (E-FWI)** — 12 synthetic seismic datasets (2.1 TB total, 2K–67K samples each) spanning interface, fault, and CO2-reservoir geological structures, purpose-built for deep-learning FWI benchmarking, plus a 3D extension.
- **Marmousi/Marmousi-2 and SEAM** velocity models — synthetic but treated as de facto standard test data for both classical and ML-based seismic imaging.
- **Public well-log corpora** — used for foundation-model pretraining, e.g., the "1.1 million North American well logs" corpus referenced in recent well-log-vision-transformer work; USGS also hosts LAS-format public well data.
- **Synthetic data generators**: much of the PINN/FNO reservoir literature (e.g., the SPE Journal FNO two-phase-flow paper, U-FNO) generates its own synthetic permeability-field ensembles (often Gaussian-random-field or multi-Gaussian geostatistical realizations) paired with numerical-solver (Eclipse/OPM/MRST) outputs as supervised training pairs — a pattern the reference repo should document as a reusable recipe (geostatistical realization generator → classical solver → neural-operator training pair).
- **USGS public data** — the USGS hosts LAS well-log archives and other public geological/geophysical data preservation resources (usgs.gov/programs/national-geological-and-geophysical-data-preservation-program), a useful discovery point for additional open field data beyond Volve.

---

## Industry Standards & Validation Context

- **SPE-PRMS (Petroleum Resources Management System)** — jointly developed by SPE, the World Petroleum Council (WPC), AAPG, and SPEE; the primary global standard for classification, estimation, and reporting of hydrocarbon reserves and resources (spe.org/en/industry/reserves; spe.org/industry/petroleum-resources-classification-system-definitions). Machine-learning-based reserve/recovery-factor estimation methods (e.g., arXiv:2010.03408 on ML for recovery-factor estimation as a de-risking tool) are being explored as inputs to PRMS-governed workflows, but public search results did not surface formal, codified SPE guidance on how ML/AI model outputs should be validated or audited *within* the PRMS framework itself — this remains an open standards gap (see below) rather than a solved problem, and is a natural area where an open-source reference repo could add real value by tracking emerging practice.
- **Model validation practice in reservoir engineering** traditionally rests on history matching against observed production/pressure data, benchmarking against SPE Comparative Solution Project cases, and expert/peer review — physics-informed ML methods are generally being validated the same way (comparison to classical-solver "ground truth" on SPE9/SPE10-style benchmarks) rather than via new AI-specific certification processes, which is itself worth documenting as a gap.
- **Regulatory considerations for AI in E&P decision-making** are still nascent industry-wide; the searched literature reflects active *technical* uncertainty-quantification work (Bayesian neural networks, MCMC-based UQ, ensemble methods) motivated by high-stakes drilling/investment decisions, but no dedicated SPE or regulatory standard specifically governing AI model auditability, explainability, or liability in reserve booking or drilling decisions was found in this research pass — flagged as an open item rather than asserted as settled.
- **OSDU** (mentioned above) is emerging as a de facto data-interoperability standard that indirectly shapes how AI tools are expected to consume/produce subsurface data across vendors.

---

## Open Challenges & Gaps

1. **Data scarcity and inherent under-sampling.** Subsurface systems are observed only through sparse wells, seismic surveys with limited resolution, and indirect proxies — unlike domains such as vision or language with abundant ground truth. Nearly every survey paper above (arXiv:2104.04764, arXiv:2308.04457) identifies this as the central limiter: most PINN/FNO "successes" are validated on synthetic data where the true field is known, and transfer to sparse real-field data (like Volve) remains harder and less consistently reported.
2. **Uncertainty quantification (UQ) for high-stakes decisions.** Drilling, completion, and investment decisions are irreversible and expensive; point predictions from neural surrogates are insufficient. Active UQ approaches include Bayesian neural networks, MCMC-based methods, ensemble Kalman-based history matching coupled to neural operators (arXiv:2406.00889), and conditional generative priors for FWI uncertainty (arXiv:2408.09975) — but consistent, industry-agreed UQ benchmarks/metrics for reservoir and seismic ML are still immature.
3. **Generalization across geological settings.** Models trained on one basin/field (e.g., Volve's North Sea Jurassic sandstone) do not reliably transfer to different rock types, structural regimes, or fluid systems. Foundation-model efforts for well logs (WLFM, arXiv:2509.18152) explicitly target cross-well/cross-basin generalization as their central value proposition, but validation is still "basin-specific" per multiple sources reviewed.
4. **Multi-physics coupling.** Real subsurface systems couple flow, geomechanics, and thermal effects (and sometimes geochemistry) simultaneously — e.g., CO2 injection changes both flow and stress state, which changes fault reactivation risk. PINN/surrogate work on flow-geomechanics coupling (poroelasticity, arXiv:2110.03049; CO2 sequestration surrogate, arXiv:2105.01334) exists but full three- or four-way coupled physics-AI models at field scale remain rare relative to single-physics (pure flow, or pure wave-equation) demonstrations.
5. **Well/point-source singularities in PINNs.** As WellPINN (arXiv:2507.09330) documents, standard PINNs handle point-source/sink boundary conditions (i.e., actual wells) poorly — a fundamental gap between academic PINN demonstrations (smooth domains, no wells) and practical reservoir engineering (everything is organized around wells).
6. **Real-time optimization for drilling.** Geosteering and drilling optimization need sub-second-to-second inference under uncertainty streamed from downhole sensors; current deep-learning geosteering work (e.g., APC-LSTM models in *SPE Drilling & Completion*) is promising but real-time, safety-critical deployment at scale is still limited compared to offline/planning-stage ML use.
7. **Compute-vs-accuracy tradeoffs and grid/resolution sensitivity.** FNO-class models show strong speedups but documented sensitivity to training/inference grid resolution (motivating multi-fidelity FNO work, ScienceDirect S0022169424000350) — "zero-shot super-resolution" claims common in the ML-PDE literature don't always hold up on heterogeneous geological fields.
8. **Reproducibility and benchmarking fragmentation.** Unlike computer vision (ImageNet) or NLP (GLUE/SuperGLUE), there is no single agreed benchmark suite spanning reservoir flow + seismic + drilling; SPE9/SPE10, OpenFWI, and Volve are each used somewhat independently by different sub-communities. This is arguably the single biggest opportunity for an open-source reference repo to add value — by aggregating and standardizing benchmark access/tooling.
9. **Standards and validation lag.** As noted above, formal PRMS-level or regulatory guidance on validating/auditing AI-derived reserve estimates and drilling recommendations has not caught up with the pace of published technical work.

---

## Roadmap: Building the Reference Hub

A phased plan for an open-source repository to become the leading public reference for "Physics AI in Oil & Gas" over 1–3 years.

### Phase 0 (Months 0–2): Foundation
- Publish this document as the seed `README`/reference page, explicitly marked as community-editable and version-controlled, with every claim traceable to a cited source (arXiv ID, DOI, or vendor URL) — never uncited assertions.
- Stand up a curated, tagged **bibliography** (BibTeX/CSL-JSON) covering all papers cited here plus an ongoing arXiv watch (physics.geo-ph, cs.LG cross-listed with subsurface/geophysics keywords).
- Stand up a **dataset registry**: Volve, SPE9/SPE10, OpenFWI, Marmousi/Marmousi-2, SEAM, with direct download links, licensing notes, and loader snippets (MRST `spe10` module, OPM `opm-tests`, `lasio` for LAS files).
- Provide minimal reproducible **starter notebooks**: OPM Flow on SPE10/SPE9, Devito FDTD on Marmousi, a DeepXDE PINN Darcy-flow example, an FNO reservoir example — each runnable end-to-end on free-tier compute.

### Phase 1 (Months 2–8): Structure & Community
- Organize the repo into clear tracks mirroring this document's sections: `reservoir/`, `seismic/`, `drilling/`, `production-pipeline/`, `datasets/`, `benchmarks/`, `standards/`.
- Build a **model architecture zoo**: reference implementations or links to canonical open-source implementations (DeepXDE for PINNs; NVIDIA PhysicsNeMo/Modulus for PINO/FNO; Devito for classical wave baselines; OPM Flow for reservoir baselines) with a consistent "physics/architecture/dataset/benchmark result" card format per model, similar to model cards in ML.
- Create a **benchmark leaderboard concept** (initially just a tracked table, not live infra) comparing reported speedups/accuracy across FNO/PINO/PINN papers on SPE9/SPE10/Volve/OpenFWI, explicitly noting where comparisons are apples-to-oranges (different resolutions, different metrics) — a curation task that has clear value and low technical risk.
- Open contribution guidelines emphasizing citation discipline (no fabricated claims, every performance number sourced) — directly addressing the field's current benchmarking fragmentation (Gap #8 above).
- Cross-post/announce in relevant venues: SEG/EAGE ML workshop mailing lists, SPE technical sections, relevant subreddits/Discord (petroleum engineering, geophysics, SciML/PINN communities), and OPM/Devito project communities to recruit domain-expert contributors, not just ML engineers.

### Phase 2 (Months 8–18): Depth & Trust
- Recruit a small **advisory review layer** (even informally — 3–5 practitioners spanning reservoir engineering, geophysics, and ML) to periodically review flagged/contested claims, given how many company-specific details (Section 3) are marketing-adjacent and need scrutiny.
- Add a **standards & validation tracker** explicitly cataloguing the gap identified above (Section 8): what SPE-PRMS does and does not say about AI-derived estimates, tracking any emerging SPE/EAGE/SEG guidance documents as they are published, rather than asserting standards that don't yet exist.
- Publish worked **hybrid-method case studies** reproducing (with attribution) the general approach of key papers — e.g., a from-scratch OPM-Flow-generated-data → FNO surrogate pipeline on SPE10, and a Devito-generated-data → DeepONet FWI pipeline on Marmousi — as pedagogical, fully reproducible references, since the field currently lacks accessible worked examples connecting classical solvers to their ML surrogates end-to-end.
- Begin tracking **commercial landscape** changes (Section 3) as a lightweight, dated changelog (product launches, acquisitions, funding rounds) rather than a static list, since this space moves quickly (e.g., Emerson's 2025 Project Beyond announcement, Novi Labs' 2025 funding round).

### Phase 3 (Year 2–3): Reference Hub Status
- Push toward becoming the **default citation/starting point** when practitioners or new researchers ask "where do I start with physics-informed ML for oil & gas" — measured by inbound links from SPE/SEG/EAGE community discussions, course syllabi, and citations in papers' "related resources" sections.
- Formalize a lightweight **governance model** (maintainers per track, e.g., a reservoir-simulation maintainer, a seismic-imaging maintainer) to sustain quality as contributions scale, avoiding both stagnation and uncurated sprawl.
- Explore **companion open benchmark infrastructure** (e.g., a hosted leaderboard with standardized SPE10/OpenFWI evaluation scripts) only once the curation-only leaderboard from Phase 1 has proven sustained community engagement — infrastructure should follow demonstrated demand, not precede it.
- Maintain the **open-questions/gaps register** (Section 9) as a living document, explicitly inviting new research to "claim" and resolve listed gaps, turning the repo into not just a reference but a lightweight research-coordination point for the field.
- Continue enforcing zero-fabrication sourcing discipline as the repo's core differentiator versus marketing-driven industry content — this is the credibility asset that makes "leading public reference" achievable at all.

---

## Sources

**Surveys & reviews**
- [Applications of physics-informed scientific machine learning in subsurface science: A survey (arXiv:2104.04764)](https://arxiv.org/abs/2104.04764)
- [A Critical Review of Physics-Informed Machine Learning Applications in Subsurface Energy Systems (arXiv:2308.04457)](https://arxiv.org/pdf/2308.04457)
- [Emerging applications of physics-informed and physics-guided machine learning in geoenergy science: A review](https://www.sciencedirect.com/science/article/abs/pii/S1007570425009608)
- [Perspectives of physics-based machine learning strategies for geoscientific applications governed by PDEs (GMD)](https://gmd.copernicus.org/articles/16/7375/2023/)
- [Survey of Deep Learning and Physics-Based Approaches in Computational Wave Imaging (arXiv:2410.08329)](https://arxiv.org/pdf/2410.08329)
- [Synergizing Deep Learning and Full-Waveform Inversion (arXiv:2502.17585)](https://arxiv.org/abs/2502.17585)
- [Applications of PINNs in fluid flow in porous media - A review study](https://www.sciencedirect.com/science/article/pii/S266654682600073X)

**Reservoir simulation: PINNs & neural operators**
- [PINN-based petroleum reservoir simulation with sparse data using domain decomposition](https://www.sciencedirect.com/science/article/pii/S1995822623002947)
- [WellPINN: Accurate Well Representation for Transient Fluid Pressure Diffusion (arXiv:2507.09330)](https://arxiv.org/pdf/2507.09330)
- [A novel reservoir simulation model based on physics informed neural networks (Physics of Fluids, AIP)](https://pubs.aip.org/aip/pof/article/36/11/116617/3322481/A-novel-reservoir-simulation-model-based-on)
- [Transfer learning-based PI-CNN for flow in porous media with time-varying controls (arXiv:2310.06319)](https://arxiv.org/pdf/2310.06319)
- [Fourier Neural Operator for Solving Subsurface Oil/Water Two-Phase Flow PDE (SPE Journal, OnePetro)](https://onepetro.org/SJ/article/27/03/1815/480880/Fourier-Neural-Operator-for-Solving-Subsurface-Oil)
- [Enhancing subsurface multiphase flow simulation with Fourier neural operator](https://www.sciencedirect.com/science/article/pii/S2405844024141345)
- [Three-phase flow reservoir simulation acceleration using U-FNO](https://www.researchgate.net/publication/369854804_Three-phase_flow_reservoir_simulation_approximation_and_acceleration_using_U-shaped_Fourier_neural_operator_U-FNO)
- [Multi-fidelity Fourier neural operator for large-scale geological carbon storage](https://www.sciencedirect.com/science/article/abs/pii/S0022169424000350)
- [A Novel A.I Enhanced Reservoir Characterization — NVIDIA Modulus PINO (arXiv:2404.14447)](https://arxiv.org/abs/2404.14447)
- [Reservoir History Matching of the Norne field with PINO forward model (arXiv:2406.00889)](https://arxiv.org/pdf/2406.00889)
- [A reduced-order derivative-informed neural operator for subsurface fluid-flow (arXiv:2509.13620)](https://arxiv.org/html/2509.13620v2)
- [Enhancing Reservoir Model History Matching with AI Surrogate and Ensemble Iterative Algorithms (SPE ATCE, OnePetro)](https://onepetro.org/SPEATCE/proceedings-abstract/24ATCE/24ATCE/D011S010R008/563801)

**Poroelasticity / geomechanics**
- [PINN simulation of multiphase poroelasticity using stress-split sequential training (arXiv:2110.03049)](https://arxiv.org/pdf/2110.03049)
- [Physics-informed deep learning for flow and deformation in poroelastic media (arXiv:2010.15426)](https://arxiv.org/pdf/2010.15426)
- [Deep-learning-based coupled flow-geomechanics surrogate model for CO2 sequestration (arXiv:2105.01334)](https://arxiv.org/pdf/2105.01334)
- [Continuous conditional GANs for poroelasticity with heterogeneous material properties (arXiv:2111.14984)](https://arxiv.org/pdf/2111.14984)

**Seismic / FWI**
- [OpenFWI: Large-scale Multi-structural Benchmark Datasets for FWI (arXiv:2111.02926)](https://arxiv.org/abs/2111.02926)
- [E-FWI: Multi-parameter Benchmark Datasets for Elastic FWI (arXiv:2306.12386)](https://arxiv.org/pdf/2306.12386)
- [Deep Neural Helmholtz Operators for 3D Elastic Wave Propagation and Inversion (arXiv:2311.09608)](https://arxiv.org/pdf/2311.09608)
- [Fourier-DeepONet for full waveform inversion](https://www.sciencedirect.com/science/article/abs/pii/S0045782523004243)
- [En-DeepONet: enrichment approach for neural operators applied to seismology](https://www.sciencedirect.com/science/article/abs/pii/S0045782523008046)
- [SG-DeepONet: Source-generalized deep operator learning for FWI (arXiv:2408.08005)](https://arxiv.org/pdf/2408.08005)
- [The Marmousi experience: Velocity model determination on a synthetic complex data set (The Leading Edge, SEG)](https://library.seg.org/doi/10.1190/1.1437051)
- [Conditional Image Prior for Uncertainty Quantification in FWI (arXiv:2408.09975)](https://arxiv.org/pdf/2408.09975)
- [Accelerating seismic inversion and UQ with efficient high-rank Hessian approximations (arXiv:2507.10804)](https://arxiv.org/pdf/2507.10804)

**Graph neural networks / production forecasting**
- [Physics-informed graph neural network for spatial-temporal production forecasting (arXiv:2209.11885)](https://arxiv.org/pdf/2209.11885)
- [Physics-informed graph neural network for predicting fluid flow in porous media](https://www.sciencedirect.com/science/article/pii/S199582262500216X)
- [Dynamic interwell connectivity analysis based on improved graph neural network](https://www.researchgate.net/publication/375585792_Dynamic_interwell_connectivity_analysis_of_multi-layer_waterflooding_reservoirs_based_on_an_improved_graph_neural_network)
- [A Graph-Based Deep-Learning Framework for Liquid Production Forecasting (SPE IOR, OnePetro)](https://onepetro.org/SPEIOR/proceedings-abstract/26IOR/26IOR/D011S002R001/797720)

**Well logs / foundation models**
- [WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation (arXiv:2509.18152)](https://arxiv.org/html/2509.18152)
- [Leveraging Time-Series Foundation Model for Subsurface Well Logs Prediction and Anomaly Detection (arXiv:2412.05681)](https://arxiv.org/pdf/2412.05681)
- [Well log foundation model: Making promptable AI models for interpretation (TGS)](https://www.tgs.com/technical-library/well-log-foundation-model-making-promptable-ai-models-for-interpretation)
- [Log ASCII Standard (Wikipedia)](https://en.wikipedia.org/wiki/Log_ASCII_standard)
- [LAS Format (USGS)](https://www.usgs.gov/programs/national-geological-and-geophysical-data-preservation-program/las-format)
- [lasio: Python library for LAS files (PyPI)](https://pypi.org/project/lasio/)

**Commercial / industry**
- [AI for Subsurface Oil & Gas Operations (NVIDIA)](https://www.nvidia.com/en-us/industries/energy/oil-gas-operations/)
- [Research and Publications Using PhysicsNeMo (NVIDIA)](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/research.html)
- [Machine learning solutions for subsurface workflows (SLB)](https://www.slb.com/products-and-services/delivering-digital-at-scale/subsurface/machine-learning)
- [Petrel AI and ML on-prem (SLB)](https://www.slb.com/products-and-services/delivering-digital-at-scale/subsurface/petrel-ai-ml-on-premises)
- [Olga dynamic multiphase flow simulator (SLB)](https://www.slb.com/products-and-services/delivering-digital-at-scale/software/olga)
- [OLGA (technology) - Wikipedia](https://en.wikipedia.org/wiki/OLGA_(technology))
- [Halliburton delivers on digital vision at Landmark Innovation Forum and Expo](https://www.halliburton.com/en/about-us/press-release/halliburton-delivers-on-digital-vision-at-landmark-innovation-forum-and-expo)
- [Emerson Acquires Paradigm](https://www.emerson.com/en/corporate/news/2017/emerson-acquires-paradigm)
- [Emerson's Expanded AI Portfolio / Project Beyond (2025)](https://www.emerson.com/en/corporate/news/2025/05-emerson-expanded-ai-portfolio-paves-the-way-for-more-optimized-autonomous-operations)
- [Cognitive AI in Oil & Gas (Beyond Limits)](https://go.beyond.ai/ai/oil-and-gas)
- [InteractivAI Deep Learning (Bluware)](https://bluware.com/data-solutions/interactivai/)
- [Bluware and Microsoft Azure OSDU-enabled AI seismic interpretation](https://azure.microsoft.com/en-us/blog/bluware-and-microsoft-azure-develop-osduenabled-interactive-ai-seismic-interpretation-solution-for-energy-super-major/)
- [Industrial DataOps: The data backbone of digital twins (Cognite)](https://www.cognite.com/en/resources/blog/industrial-dataops-the-data-backbone-of-digital-twins)
- [Transform oil and gas with Industrial DataOps (Cognite)](https://www.cognite.com/en/industries/oil-and-gas)
- [Aize | The next generation of digital twin](https://www.aize.io/)
- [Aize signs agreement with ExxonMobil Guyana and SBM Offshore](https://www.aize.io/blog/aize-agreement-exxonmobil-and-sbm)
- [Novi Labs - Oil & Gas Data Analytics](https://novilabs.com/)
- [Novi Labs Raises $35 Million from Invictus Growth Partners](https://novilabs.com/news/novi-labs-raises-35-million-from-invictus-growth-partners/)
- [Forget Type Curves: Novi's AI Offers Window into Undrilled Wells (Hart Energy)](https://www.hartenergy.com/exclusives/forget-type-curves-novis-ai-offers-window-undrilled-wells-213468/)

**Solvers, benchmarks & datasets**
- [The Open Porous Media Flow Reservoir Simulator (arXiv:1910.06059)](https://arxiv.org/abs/1910.06059)
- [OPM | The Open Porous Media Initiative](https://opm-project.org/)
- [OPM tests repository (SPE10 documentation)](https://github.com/OPM/opm-tests/blob/master/spe10/SPE10.md)
- [SPE10: Access to the SPE10 benchmark case (MRST, SINTEF)](https://www.sintef.no/contentassets/2551f5f85547478590ceca14bc13ad51/spe10.html)
- [Volve field data set download (Equinor)](https://www.equinor.com/energy/volve-data-sharing)
- [Volve Field Open Dataset - Open Well Data Sources](https://www.andymcdonald.scot/data/volve/)
- [Equinor's Volve Field Test Data (Energistics)](https://energistics.org/equinors-volve-field-test-data)
- [DeepXDE: A library for scientific machine learning and physics-informed learning (GitHub, lululxvi)](https://github.com/lululxvi/deepxde)
- [SEG Events](https://seg.org/events/)
- [EAGE Workshop on Enhancing Subsurface Practices using AI/ML (EarthDoc)](https://www.earthdoc.org/content/proceedings/eage-workshop-on-machine-learning-for-geoscience)

**Standards**
- [SPE Petroleum Reserves and Resources Definitions](https://www.spe.org/en/industry/reserves/)
- [Petroleum Resources Classification System Definitions (SPE)](https://www.spe.org/industry/petroleum-resources-classification-system-definitions/)
- [Machine learning for recovery factor estimation of an oil reservoir (arXiv:2010.03408)](https://arxiv.org/pdf/2010.03408)

**Drilling / uncertainty**
- [AI at the Helm: Quantifying the Next Value Revolution in Upstream Oil and Gas (JPT/SPE)](https://jpt.spe.org/twa/ai-at-the-helm-quantifying-the-next-value-revolution-in-upstream-oil-and-gas)
- [From Seismic Data to Drilling Decisions: How AI is Reshaping Oil and Gas Exploration (JPT/SPE)](https://jpt.spe.org/twa/from-seismic-data-to-drilling-decisions-how-ai-is-reshaping-oil-and-gas-exploration)
- [TADI: Tool-Augmented Drilling Intelligence via Agentic LLM Orchestration (arXiv:2605.00060)](https://arxiv.org/pdf/2605.00060)
- [Early signs of stuck pipe detection based on Crossformer (arXiv:2503.07440)](https://arxiv.org/pdf/2503.07440)

---

*Compiled through iterative web research (September 2026). This document is intended as a living, editable seed for an open-source repository — contributors should verify, update, and expand every section with primary sources, and flag any claim that cannot be traced to a citation.*
