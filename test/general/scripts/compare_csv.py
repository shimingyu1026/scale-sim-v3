#!/usr/bin/env python3

import argparse
import csv
import math
import sys


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [trim_trailing_empty_cells([cell.strip() for cell in row]) for row in csv.reader(handle)]


def trim_trailing_empty_cells(row):
    trimmed = list(row)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()
    return trimmed


def maybe_float(value):
    try:
        return float(value)
    except ValueError:
        return None


def compare_rows(expected_rows, actual_rows, numeric=False, allow_extra_columns=False,
                 rel_tol=1e-12, abs_tol=1e-12):
    if len(expected_rows) != len(actual_rows):
        return f"Row count mismatch: expected {len(expected_rows)}, got {len(actual_rows)}"

    for row_idx, (expected_row, actual_row) in enumerate(zip(expected_rows, actual_rows), start=1):
        if allow_extra_columns:
            if len(actual_row) < len(expected_row):
                return (
                    f"Row {row_idx} has too few columns: expected at least {len(expected_row)}, "
                    f"got {len(actual_row)}"
                )
        elif len(expected_row) != len(actual_row):
            return (
                f"Row {row_idx} column count mismatch: expected {len(expected_row)}, "
                f"got {len(actual_row)}"
            )

        compare_len = len(expected_row)
        for col_idx in range(compare_len):
            expected_cell = expected_row[col_idx]
            actual_cell = actual_row[col_idx]

            if numeric and row_idx > 1:
                expected_num = maybe_float(expected_cell)
                actual_num = maybe_float(actual_cell)
                if expected_num is not None and actual_num is not None:
                    if not math.isclose(expected_num, actual_num, rel_tol=rel_tol, abs_tol=abs_tol):
                        return (
                            f"Numeric mismatch at row {row_idx}, col {col_idx + 1}: "
                            f"expected {expected_cell}, got {actual_cell}"
                        )
                    continue

            if expected_cell != actual_cell:
                return (
                    f"Mismatch at row {row_idx}, col {col_idx + 1}: "
                    f"expected {expected_cell!r}, got {actual_cell!r}"
                )

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--allow-extra-columns", action="store_true")
    parser.add_argument("--rel-tol", type=float, default=1e-12)
    parser.add_argument("--abs-tol", type=float, default=1e-12)
    args = parser.parse_args()

    expected_rows = read_csv(args.expected)
    actual_rows = read_csv(args.actual)
    error = compare_rows(
        expected_rows=expected_rows,
        actual_rows=actual_rows,
        numeric=args.numeric,
        allow_extra_columns=args.allow_extra_columns,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    if error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
