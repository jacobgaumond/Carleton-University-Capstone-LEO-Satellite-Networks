"""Extract rows for one ICAO24 address from an ADS-B history CSV."""

import argparse
import csv
import re
from pathlib import Path


DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "history"
    / "adsb_trajectory_data.csv"
)


def _parse_icao24(value: str) -> str:
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", value):
        raise argparse.ArgumentTypeError(
            "ICAO24 must be exactly six hexadecimal characters."
        )
    return value.upper()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract rows for one ICAO24 address from an ADS-B history CSV."
    )
    parser.add_argument(
        "--icao24",
        required=True,
        type=_parse_icao24,
        help="Six-character hexadecimal ICAO24 address.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: next to the input, named for the ICAO24).",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.with_name(
            f"{input_path.stem}_icao24_{args.icao24}{input_path.suffix}"
        )
    )
    if input_path == output_path:
        parser.error("Output path must not be the same as the input path.")

    with input_path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "icao24" not in reader.fieldnames:
            parser.error("Input CSV must contain an 'icao24' column.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
            writer.writeheader()
            match_count = 0
            for row in reader:
                address = row.get("icao24")
                if address is not None and address.strip().upper() == args.icao24:
                    writer.writerow(row)
                    match_count += 1

    print(f"Saved {match_count} matching rows to {output_path}")


if __name__ == "__main__":
    main()
