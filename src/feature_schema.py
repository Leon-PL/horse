"""Shared categorical domains and required numeric feature validation.

Categorical columns are declared from their source fields, never inferred from
a feature-name prefix or an observed numeric range.
"""
import pandas as pd


CATEGORICAL_FIELDS = ("going", "race_class", "race_type", "region", "sex", "surface")
FIXED_DOMAINS = {
    "race_type": ("Flat", "Hurdle", "Chase"),
    "sex": ("Male", "Female"),
    "season": ("Winter", "Spring", "Summer", "Autumn"),
}


def categorical_columns(frame):
    """Explicit dummy names for the categories represented by prepared rows."""
    columns = set()
    for field in CATEGORICAL_FIELDS:
        if field in frame:
            labels = set(frame[field].dropna().unique()) | set(FIXED_DOMAINS.get(field, ()))
            columns.update(f"{field}_{label}" for label in labels)
    if "race_date" in frame:
        columns.update(f"season_{label}" for label in FIXED_DOMAINS["season"])
    return sorted(columns)


def complete_categorical_columns(frame, expected, declared):
    """Complete absent declared dummies; continuous measurements stay untouched."""
    missing = (set(expected) & set(declared)) - set(frame.columns)
    if not missing:
        return frame
    zeros = pd.DataFrame(0, index=frame.index, columns=sorted(missing), dtype="int64")
    return pd.concat([frame, zeros], axis=1)


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
