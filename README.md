# gic-tools

Python tools for modelling geomagnetically induced currents (GICs) in power networks. The grid model combines network connectivity, resistances, grounding information, and geoelectric-field inputs to calculate currents in transformers and power lines.

The grid module is a further development of [GEOMAGICA](https://github.com/bairaelyn/GEOMAGICA), originally developed by R. Bailey at the Conrad Observatory, GeoSphere Austria.

## Workflow

1. Load a power-network configuration into `PowerGrid`.
2. Prepare northward and eastward electric-field arrays with matching UTC timestamps.
3. Calculate network currents with `calc_gic_in_grid()`.
4. Extract transformer and power-line time series.
5. Package both outputs in one xarray dataset, inspect them, and save to NetCDF.

## Getting started

Clone the repository and work from its root:

```bash
git clone https://github.com/bairaelyn/gic-tools.git
cd gic-tools
```

For the grid calculation and new result-export workflow, install:

```bash
python -m pip install numpy xarray netCDF4
```

Matplotlib is needed for the plotting example below. These dependencies cover this workflow, not necessarily every module in the repository. Package installation commands and supported Python versions should be confirmed against the repository's packaging configuration.

## Network configuration

```python
import configparser
from gictools.grid import PowerGrid

config = configparser.ConfigParser()
config.read('examples/config_example.ini')

grid = PowerGrid()
grid.load_Grid(config)
```

Use a configuration and network files appropriate to your grid. The loader reads these configuration fields:

| Section | Fields |
|---|---|
| `ConductivityModel` | `ModelNumber`, `Source`, `Resistivities`, `Thicknesses` |
| `PowerGrid` | `NetworkPath`, `TransformerPath`, `ConnectionsPath`, `HV-LV-Levels` |
| `BoundingBox` | `CellSpacing-X`, `CellSpacing-Y`, `NCells-X`, `NCells-Y`, `NorthBoundary`, `SouthBoundary`, `EastBoundary`, `WestBoundary` |

The network loader expects specific tab-separated file layouts. Follow the loader's column definitions or a compatible existing configuration when preparing new files. `HV-LV-Levels` is split using the literal `\t` delimiter in the supplied loader.

The `per_phase` option affects transformer resistance preparation and the current interpretation. Record the actual current convention in the output metadata.

## Calculate GICs

`En` and `Ee` are dictionaries keyed by conductivity-model identifier. Each value contains electric-field samples with time as the first axis, followed by the spatial axes expected by the selected calculation path. Supply matching arrays and timestamps, and confirm electric-field units against the input preparation code.

```python
# Create a test geoelectric field
En, Ee = gictools.efield.make_test_efield(PowerGrid, 1000, 1000, n_timesteps=1)
En, Ee = {}, {}
En['39'] = np.array([np.full((PowerGrid.x_ncells, PowerGrid.y_ncells), x) for x in En_t])
Ee['39'] = np.array([np.full((PowerGrid.x_ncells, PowerGrid.y_ncells), x) for x in Ee_t])

# Calculate GICs in whole grid:
grid_results = grid.calc_gic_in_grid(En, Ee)

# Derive further results:
# (set print_results=True to print a table of output)
transformer_results = grid.calc_gic_in_transformers(grid_results)
line_results = grid.calc_gic_in_power_lines(grid_results)
substation_results =  PowerGrid.calc_gic_in_substations(results, print_results=True)
```

The grid calculation returns intermediate arrays for each timestep. The extraction methods return dictionaries of current series in amperes. Keys ending in `_GICX` and `_GICY` contain contributions from northward and eastward electric fields, respectively.

For multiprocessing (`threading=True`), run script entry points under an `if __name__ == "__main__":` guard.

### Calculation assumptions

- The supplied fast solver samples the first spatial cell of the first selected conductivity model. Use it for a spatially uniform field; supplying spatially varying arrays does not make this path spatially resolved.
- The interpolation path is a separate implementation and should be checked for compatibility and consistent sign conventions before use.
- Line ID `0` denotes transformer connections and is excluded from power-line output.
- Repeated power-line IDs represent the same line in this workflow. Extraction and output metadata retain the last connection associated with each ID. This does not remove duplicate records from the solver's network calculation.
- In the supplied transformer extraction code, remove the trailing comma from the assignment to `codes` so the forward connection lookup uses a string rather than a tuple.

## Combined xarray output

```python
from gictools.results import (
    gic_results_to_xarray, print_gic_results, save_gic_results,
)

ds = gic_results_to_xarray(
    transformer_results, line_results, times, grid,
    metadata={
        "grid_version": "network-configuration-version",
        "code_version": "software-version",
        "code_commit": "git-commit",
        "conductivity_model": "39",
        "electric_field_mode": "spatially_uniform",
        "input_source": "electric-field input description",
        "current_definition": "document neutral or per-phase convention",
        "transformer_positive_current_direction": "document transformer convention",
        "line_positive_current_direction": "line_from_node to line_to_node",
        "timestamp_convention": "instantaneous value at the stated timestamp",
    },
)
```

Replace the example metadata with the conventions and identifiers used for your run. The line direction shown matches the supplied fast solver. Grid version identifies the network configuration; code version and commit identify the software. Dataset creation time is recorded automatically in UTC.

Supply a nonempty, strictly increasing UTC time array corresponding exactly to the input samples. The dataset stores timestamps as `datetime64[ns]`, with UTC documented in attributes. Convert any other timezone to UTC before passing timestamps; NumPy datetime arrays do not retain timezone information.

### Dataset structure

| Variable | Dimensions | Units |
|---|---|---|
| `transformer_gic_total` | `time, transformer` | A |
| `transformer_gic_from_north` | `time, transformer` | A |
| `transformer_gic_from_east` | `time, transformer` | A |
| `line_gic_total` | `time, line` | A |
| `line_gic_from_north` | `time, line` | A |
| `line_gic_from_east` | `time, line` | A |

Coordinates include UTC time, transformer IDs, unique line IDs, line endpoint node names, and endpoint latitude and longitude. Empty transformer placeholders are excluded. Transformer location coordinates are not yet included in this interface.

Endpoint coordinates describe the connection endpoints, not the full routed line geometry. Dataset attributes contain run provenance; variable attributes contain units, descriptions, and current conventions. This schema is not a claim of full CF compliance.

### Inspect results

```python
print_gic_results(ds)                          # Both categories, first timestep
print_gic_results(ds, data="transformer")      # Transformers only
print_gic_results(ds, timestep=10, data="line")

print(ds["transformer"].values)
print(ds["line"].values)
```

### Save and reload

```python
import xarray as xr

filepath = save_gic_results(ds, "output/gic_results.nc")
reloaded = xr.load_dataset(filepath)
```

The save function creates missing directories, applies a `.nc` extension, and writes both categories to one NetCDF file. Floating-point variables use lossless compression at level 4. Existing files at the selected path are overwritten.

## Plot a transformer time series

```python
import matplotlib.pyplot as plt

transformer_id = str(ds["transformer"].values[0])
series = ds["transformer_gic_total"].sel(transformer=transformer_id)

fig, ax = plt.subplots()
ax.plot(series["time"].values, series.values)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("GIC (A)")
ax.set_title(transformer_id)
fig.autofmt_xdate()
plt.show()
```

Select a line similarly with `ds["line_gic_total"].sel(line="your-line-id")`.

## Validation with Horton et al. (2012) example grid

A full example of the code running on the [Horton et al. (2012)](https://doi.org/10.1109/TPWRD.2012.2206407) example grid is provided in `examples/calculate_gics.py`. This code uses some Horton-specific geometry to calculate the GICs between stations.

The output of the script, which can be compared to Table VII in the Horton paper, is as follows:

### Substation currents

| Name | GIC_N | GIC_E | GIC_TOT |
|---|---:|---:|---:|
| Sub1 | -0.00 | -0.00 | -0.00 |
| Sub2 | 115.63 | -189.30 | -73.68 |
| Sub3 | 139.86 | -109.56 | 30.30 |
| Sub4 | 19.98 | -124.59 | -104.61 |
| Sub5 | -279.07 | -65.51 | -344.59 |
| Sub6 | -57.31 | 354.11 | 296.81 |
| Sub8 | 60.90 | 134.86 | 195.76 |

Small discrepancies are due to the Sub6 -> Sub7 -> Sub8 line being treated as a Sub6 -> Sub8 line.

## Questions and contributions

Use the [repository issue tracker](https://github.com/bairaelyn/gic-tools/issues) for bug reports and questions. Include the relevant configuration, input shapes, calculation options, traceback, and a small reproducible example where possible.

Before distributing code or network data, confirm the applicable repository license and data permissions. License and citation files were not available for verification when this draft was prepared.
