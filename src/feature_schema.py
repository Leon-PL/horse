"""Required numeric model-feature validation."""
import pandas as pd


def require_numeric_features(frame, names, *, context):
    """Missing values are valid; missing, duplicated or nonnumeric columns are not."""
    names = list(names)
    missing = [name for name in names if name not in frame]
    duplicated = sorted(set(frame.columns[frame.columns.duplicated()]) & set(names))
    invalid = [name for name in names if name in frame and name not in duplicated
               and (not pd.api.types.is_numeric_dtype(frame[name])
                    or pd.api.types.is_complex_dtype(frame[name]))]
    errors = []
    for label, values in (("missing", missing), ("duplicated", duplicated), ("nonnumeric", invalid)):
        if values:
            errors.append(f"{label}: {', '.join(values[:20])}" + (" ..." if len(values) > 20 else ""))
    if errors:
        raise ValueError(f"{context} requires its complete numeric feature schema ({'; '.join(errors)})")
