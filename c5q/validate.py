#!/usr/bin/env python3
"""
C5Q Dataset Validation CLI

Comprehensive validation tool for the C5 Quantum Logic Matrix dataset
with detailed reporting and integrity checking.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from .io import comprehensive_dataset_validation, generate_dataset_checksum


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def print_validation_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary of validation results."""
    print("\n" + "="*60)
    print("C5Q DATASET VALIDATION SUMMARY")
    print("="*60)

    # Dataset info
    info = results["dataset_info"]
    print(f"📊 Dataset: {info['total_events']:,} events, {info['total_columns']} columns")
    print(f"💾 Memory usage: {info['memory_usage_mb']:.2f} MB")

    # Checksum info
    checksum = results["checksum"]
    print(f"🔐 SHA256: {checksum['sha256'][:16]}...")
    print(f"📁 File size: {checksum['file_size_bytes']:,} bytes")

    print("\n🧪 VALIDATION TESTS:")
    print("-" * 40)

    # Test results
    for test_name, test_result in results["validation_tests"].items():
        status = test_result["status"]
        error_count = len(test_result["errors"])

        status_emoji = "✅" if status == "PASSED" else "❌"
        test_display = test_name.replace("_", " ").title()

        print(f"{status_emoji} {test_display}: {status}")
        if error_count > 0:
            print(f"   └── {error_count} error(s) found")

    # Overall result
    print("\n" + "="*60)
    if results["validation_passed"]:
        print("🎉 OVERALL RESULT: VALIDATION PASSED")
        print("   Dataset is ready for analysis and modeling.")
    else:
        total_errors = results["error_summary"]["total_errors"]
        print(f"❌ OVERALL RESULT: VALIDATION FAILED")
        print(f"   {total_errors} total error(s) found.")
        print("   Review errors before proceeding with analysis.")

    print("="*60 + "\n")


def print_detailed_errors(results: Dict[str, Any], max_errors: int = 20) -> None:
    """Print detailed error information."""
    if results["validation_passed"]:
        return

    print("🔍 DETAILED ERROR REPORT:")
    print("-" * 40)

    error_count = 0
    for test_name, test_result in results["validation_tests"].items():
        if test_result["status"] == "FAILED":
            errors = test_result["errors"]
            test_display = test_name.replace("_", " ").title()

            print(f"\n❌ {test_display} Errors ({len(errors)} total):")

            for i, error in enumerate(errors[:10]):  # Show first 10 per test
                print(f"   {i+1}. {error}")
                error_count += 1

                if error_count >= max_errors:
                    remaining = sum(len(t["errors"]) for t in results["validation_tests"].values())
                    remaining -= error_count
                    if remaining > 0:
                        print(f"   ... and {remaining} more errors")
                    return

            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors in this category")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Validate C5 Quantum Logic Matrix dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m c5q.validate --csv data/c5_Matrix_binary.csv
  python -m c5q.validate --csv data/c5_Matrix_binary.csv --output artifacts/validation/
  python -m c5q.validate --csv data/c5_Matrix_binary.csv --verbose --no-checksum
        """
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to c5_Matrix_binary.csv file"
    )

    parser.add_argument(
        "--output",
        help="Output directory for validation results (optional)"
    )

    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip saving checksum file"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only show errors and final result"
    )

    parser.add_argument(
        "--checksum-only",
        action="store_true",
        help="Only generate checksum, skip validation"
    )

    args = parser.parse_args()

    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)

    # Validate file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: Dataset file not found: {csv_path}")
        sys.exit(1)

    try:
        if args.checksum_only:
            # Generate checksum only
            print("🔐 Generating dataset checksum...")
            checksum_data = generate_dataset_checksum(args.csv)

            print(f"✅ SHA256: {checksum_data['sha256']}")
            print(f"📁 File: {checksum_data['file_name']}")
            print(f"📊 Size: {checksum_data['file_size_bytes']:,} bytes")

            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                checksum_file = output_path / f"{csv_path.stem}_checksum.json"

                with open(checksum_file, "w") as f:
                    json.dump(checksum_data, f, indent=2)

                print(f"💾 Checksum saved to: {checksum_file}")

        else:
            # Run full validation
            if not args.quiet:
                print("🧪 Starting comprehensive dataset validation...")

            results = comprehensive_dataset_validation(
                args.csv,
                save_checksum=not args.no_checksum
            )

            # Print results
            if not args.quiet:
                print_validation_summary(results)
                print_detailed_errors(results)
            else:
                # Quiet mode - only show final result
                if results["validation_passed"]:
                    print("✅ VALIDATION PASSED")
                else:
                    total_errors = results["error_summary"]["total_errors"]
                    print(f"❌ VALIDATION FAILED ({total_errors} errors)")
                    for error in results["error_summary"]["error_details"][:5]:
                        print(f"   • {error}")

            # Save detailed results if output directory specified
            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)

                results_file = output_path / f"{csv_path.stem}_validation.json"
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)

                if not args.quiet:
                    print(f"💾 Detailed results saved to: {results_file}")

            # Exit with error code if validation failed
            if not results["validation_passed"]:
                sys.exit(1)

    except Exception as e:
        print(f"❌ Validation failed with exception: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()