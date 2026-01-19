# Denmark Cost-of-Living Visualizations (2024)

This package splits the pipeline into:
- `data_io.py` — parsing CSVs, building the master municipality table, JSON serialization/cache
- `plotting.py` — all figure creation (PNGs + combined PDF)
- `run_pipeline.py` — CLI entry point / flow control
- `utils.py`, `config.py` — shared helpers

## Expected input files (under `<root>/data/`)
- `INDKF132.csv` (income)
- `FOLK1A.csv` (population)
- `PSKAT.csv` (tax rate)
- `Huslejestatistik_2024_TABLE7.csv` (rent table)
- `BM010.csv` (property prices)
- `LIGELB1_residence.csv`, `LIGELB1_workplace.csv` (earnings)
- `nuts_muni_region.csv` (municipality → region mapping)
- Optional: `LABY32.csv`, `LABY45.csv`, `HUS1.csv`
- Optional maps: `DAGI_V1_Kommuneinddeling_TotalDownload_gpkg_Current_507.gpkg` (must contain column `navn`)

All of the above should be placed in **`<root>/data/`**.

## Run
Parse CSVs (default) and generate figures:
```bash
python run_pipeline.py
```

Use cached JSON instead of parsing (after you have generated it once):
```bash
python run_pipeline.py --use-json
```

Force rebuild of the JSON cache:
```bash
python run_pipeline.py --rebuild-json
```

## Outputs
- `./figures/*.png`
- `./figures/all_figures_2024.pdf`
- `./cache/parsed_data_2024.json`

