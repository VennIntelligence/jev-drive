# Recovery rows: probe gates (plans/2026-10-21-recovery-rows.md)

- **G1 zero perturbation = on1_PAIH** (11 probe scenes, 243 rows): row files equal True, ego True, label True, front tokens 1.0000, teacher True -> **pass**
- **G2 label in raster / lateral acceleration** (243 rows): inside 1.0000 of 243 rows whose twin keeps its own label inside (all raster rows 1.0000, plain twins 1.0000); a_lat <= max(4, twin) 1.0000 (<= 4: 1.0000, twins 1.0000; p50 / p90 / p99 {'50': 0.62614399485855, '90': 1.6767144087489882, '99': 3.32039535723553}) -> **pass**
- **G3 render sanity** (243 rows): R1 median 0.0150 m, over 0.053, wrong model 1.000 (pass); NTV0 plan vs its plan on the plain twin: 4 s length ratio median 0.993, world 4 s endpoint distance median 0.95 m (p90 2.29), ego-frame max diff median 1.22 m; plan endpoint closer to the log than the ego on 0.9117647058823529 of 68 rows with |offset| >= 0.5 m -> **pass**

**Probe gates: all pass**
