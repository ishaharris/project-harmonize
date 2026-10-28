# Experimental-coordinate test fixtures

`3K82.cif` and `3VVX.cif` retain the deposited polymer sequence/mapping tables and
chain-A Cα atom rows from the public RCSB mmCIF records. They test modified residues,
missing coordinates, alternate locations, and author numbering. They are not predictions.

Source URLs and full source checksums are recorded in `inputs/sources.lock.json`.
The six-protein experimental Cα snapshot in `inputs/references.json` permits offline
notebook validation. Synthetic perturbations are generated only under `runs/validation`.
