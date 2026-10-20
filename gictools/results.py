#!/usr/bin/env python
"""
--------------------------------------------------------------------
gictools.results

Functions for handling the results provided by gictools.grid and the
functions calc_gic_in_transformers() and calc_gic_in_power_lines().

Created 2026 by R Bailey, Conrad Observatory, GeoSphere Austria
Last updated Oct 2026.
--------------------------------------------------------------------
"""

import numpy as np
import xarray as xr
from datetime import datetime, timezone, timedelta
from pathlib import Path


def gic_results_to_xarray(transformer_results, line_results, times, grid, metadata=None):
    """
    Combine transformer and power-line GIC time series into an xarray Dataset.
    See also PowerGrid.calc_gic_in_transformers() and
    PowerGrid.calc_gic_in_power_lines().

    Parameters
    ----------
    transformer_results : dict
        Output from calc_gic_in_transformers().
        Keys are transformer IDs, with '_GICX' and '_GICY' suffixes
        for northward- and eastward-electric-field contributions.
        Values are current time series in amperes.
        Empty transformer entries are excluded.
    line_results : dict
        Output from calc_gic_in_power_lines().
        Keys are line IDs, with '_GICX' and '_GICY' suffixes
        for northward- and eastward-electric-field contributions.
        Values are current time series in amperes.
    times : array-like, shape=(n_timesteps,)
        UTC timestamps corresponding to the calculation timesteps.
        Must be nonempty, one-dimensional, valid, unique, and increasing.
        Converted to numpy datetime64[ns]; timezone information is not
        retained in the array.
    grid : gictools.grid.PowerGrid
        Grid used for the calculation. Provides line IDs, connection
        indices, endpoint node names, and geographic coordinates.
    metadata : dict, optional
        Run metadata with string keys and string or numeric values.
        Includes grid and code versions, input sources, conductivity
        model, calculation mode, and current and timestamp conventions.
        Unspecified standard fields are recorded as 'unknown'.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dimensions 'time', 'transformer', and 'line'.
        Current variables, in amperes:
          'transformer_gic_total', 'line_gic_total'
          'transformer_gic_from_north', 'line_gic_from_north'
          'transformer_gic_from_east', 'line_gic_from_east'
        Includes transformer and line IDs, line endpoint node names,
        and endpoint latitudes and longitudes.
    """

    times = np.asarray(times, dtype="datetime64[ns]")
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("Provide a nonempty, one-dimensional UTC time array.")
    if np.isnat(times).any() or np.any(times[1:] <= times[:-1]):
        raise ValueError("Timestamps must be valid, unique, and increasing.")

    # Exclude component keys and unused transformer placeholders.
    transformer_ids = [
        key for key, values in transformer_results.items()
        if not key.endswith(("_GICX", "_GICY")) and len(values) > 0
    ]
    if not transformer_ids:
        raise ValueError("No transformer results were found.")

	# Match calc_gic_in_power_lines(): retain the last occurrence of each ID.
    id_to_index = {}
    for i in range(grid.nconnects):
        line_id = str(grid.id_lines[i])
        if line_id != "0":
            id_to_index[line_id] = i

    line_ids = list(id_to_index)
    indices = list(id_to_index.values())

    variables = {}
    descriptions = {}
    for prefix, dimension, results, ids in [
        ("transformer", "transformer", transformer_results, transformer_ids),
        ("line", "line", line_results, line_ids),
    ]:
        for quantity, suffix, description in [
            ("gic_total", "", "Total signed current"),
            ("gic_from_north", "_GICX", "Current contribution from northward electric field"),
            ("gic_from_east", "_GICY", "Current contribution from eastward electric field"),
        ]:
            columns = []
            for asset_id in ids:
                key = asset_id + suffix
                if key not in results:
                    raise ValueError(f"Missing {key}")

                values = np.asarray(results[key], dtype=float)
                if values.shape != (len(times),):
                    raise ValueError(f"{key} has shape {values.shape}; expected ({len(times)},).")
                columns.append(values)

            name = f"{prefix}_{quantity}"
            data = np.column_stack(columns) if columns else np.empty((len(times), 0))
            variables[name] = (("time", dimension), data)
            descriptions[name] = description

    from_indices = np.asarray(grid.nodefrom, dtype=int)[indices]
    to_indices = np.asarray(grid.nodeto, dtype=int)[indices]
    latitudes = np.asarray(grid.latitudes)
    longitudes = np.asarray(grid.longitudes)

    metadata = {} if metadata is None else dict(metadata)

    # NetCDF attributes should use simple, serializable values.
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise ValueError("Metadata must have string keys and string or numeric values.")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    attrs = {
        "title": "Transformer and power-line GIC results",
        "grid_version": "unknown",
        "code_version": "unknown",
        "code_commit": "unknown",
        "conductivity_model": "unknown",
        "electric_field_mode": "unknown",
        "input_source": "unknown",
        "current_definition": "unknown",
        "transformer_positive_current_direction": "unknown",
        "line_positive_current_direction": "unknown",
        "time_reference": "UTC",
        **metadata,
        "schema_version": "1.0",
        "date_created": created,
        "history": f"{created}: Created with gic_results_to_xarray.",
    }

    ds = xr.Dataset(
        data_vars=variables,
        coords={
            "time": times,
            "transformer": np.asarray(transformer_ids, dtype=str),
            "line": np.asarray(line_ids, dtype=str),
            "line_from_node": ("line", np.asarray([str(grid.idx_to_st[i]) for i in from_indices], dtype=str)),
            "line_to_node": ("line", np.asarray([str(grid.idx_to_st[i]) for i in to_indices], dtype=str)),
            "line_from_latitude": ("line", latitudes[from_indices]),
            "line_from_longitude": ("line", longitudes[from_indices]),
            "line_to_latitude": ("line", latitudes[to_indices]),
            "line_to_longitude": ("line", longitudes[to_indices]),
        },
        attrs=attrs,
    )

    ds["time"].attrs.update(standard_name="time", timezone="UTC")
    for name, description in descriptions.items():
        category = "transformer" if name.startswith("transformer_") else "line"
        ds[name].attrs.update(
            units="A",
            long_name=description,
            current_definition=attrs["current_definition"],
            positive_current_direction=attrs[f"{category}_positive_current_direction"],
        )

    for endpoint in ("from", "to"):
        ds[f"line_{endpoint}_latitude"].attrs.update(standard_name="latitude", units="degrees_north")
        ds[f"line_{endpoint}_longitude"].attrs.update(standard_name="longitude", units="degrees_east")

    return ds


def print_gic_results(ds, timestep=0, data="both"):
    """
    Print transformer and/or power-line currents at one timestep.

    Prints northward- and eastward-field contributions and total
    current in amperes, with the UTC timestamp.

    Parameters
    ----------
    ds : xarray.Dataset
        Output from gic_results_to_xarray().
    timestep : int, default=0
        Index of the timestep to print.
    data : str, default='both'
        Results to print: 'transformer', 'line', or 'both'.
    """

    if data not in ("transformer", "line", "both"):
        raise ValueError("data must be 'transformer', 'line', or 'both'.")

    selected = ds.isel(time=timestep)
    timestamp = str(selected["time"].values).replace("T", " ")
    categories = ("transformer", "line") if data == "both" else (data,)

    for category in categories:
        title = "TRANSFORMER CURRENTS" if category == "transformer" else "POWER LINE CURRENTS"
        print(f"\n{title} — {timestamp} UTC")
        print("-" * len(title))
        print(f"{'ID':<24}\t{'GIC_N [A]':>12}\t{'GIC_E [A]':>12}\t{'GIC_TOT [A]':>12}")

        for asset_id in selected[category].values:
            row = selected.sel({category: asset_id})
            north = row[f"{category}_gic_from_north"].item()
            east = row[f"{category}_gic_from_east"].item()
            total = row[f"{category}_gic_total"].item()
            print(f"{str(asset_id):<24}\t{north:12.2f}\t{east:12.2f}\t{total:12.2f}")


def rename_transformers(gic_trans, transformer_codes, ignore_duplicates=True):
    """
    Rename transformer IDs, preserving optional component suffixes.

    Parameters
    ----------
    gic_trans : dict
        Transformer current series keyed by ID, optionally including
        '_GICX' and '_GICY' component keys.
    transformer_codes : dict
        Original IDs mapped to new transformer designations.
        Leading and trailing whitespace is removed from designations.
    ignore_duplicates : bool, default=True
        If True, later entries overwrite duplicate output keys.
        Otherwise, duplicate output keys raise ValueError.

    Returns
    -------
    renamed : dict
        Renamed current series. Unmapped input keys are printed
        and excluded. Values reference the original series.

    Raises
    ------
    ValueError
        If a designation is empty or not a string, or an output key
        repeats when ignore_duplicates=False.
    KeyError
        If a mapped original ID is absent from gic_trans.
    """
    renamed = {}

    for old_id, new_id in transformer_codes.items():
        if not isinstance(new_id, str) or not new_id.strip():
            raise ValueError(f"Provide a transformer designation for {old_id}.")
        new_id = new_id.strip()

        for suffix in ("", "_GICX", "_GICY"):
            old_key = old_id + suffix
            if old_key not in gic_trans:
                if suffix == "":
                    raise KeyError(f"Transformer not found: {old_id}")
                continue

            new_key = new_id + suffix
            if new_key in renamed and ignore_duplicates == False:
                raise ValueError(f"Duplicate output key: {new_key}")
            renamed[new_key] = gic_trans[old_key]

    unmapped = set(gic_trans) - {
        old_id + suffix
        for old_id in transformer_codes
        for suffix in ("", "_GICX", "_GICY")
    }
    if unmapped:
        print(f"Unmapped result keys: {sorted(unmapped)}")

    return renamed


def save_gic_results(ds, filepath):
    """
    Save GIC results to NetCDF, overwriting an existing file.
    Requires netCDF4. Floating-point variables use compression
    level 4. Coordinates and metadata are included.

    Parameters
    ----------
    ds : xarray.Dataset
        Output from gic_results_to_xarray().
    filepath : str or pathlib.Path
        Output path. The suffix is set to '.nc' if needed,
        and missing parent directories are created.

    Returns
    -------
    filepath : pathlib.Path
        Path to the saved file.
    """

    filepath = Path(filepath)
    if filepath.suffix.lower() != ".nc":
        filepath = filepath.with_suffix(".nc")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name in ds.data_vars
        if ds[name].dtype.kind == "f"
    }

    ds.to_netcdf(filepath, engine="netcdf4", encoding=encoding)
    return filepath



