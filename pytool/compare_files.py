import sys
import os

def compare_files_print_diff_lines(file1_path, file2_path):
    """
    Compares two text files line by line.

    Prints only the line numbers where differences occur.
    Also reports if files have different lengths.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.

    Returns:
        bool: True if files are identical, False otherwise.
              Prints differing line numbers to stdout if not identical.
              Prints error details to stderr if files cannot be opened.
    """
    any_difference_found = False
    differing_lines = []
    length_mismatch_line = None
    f1_basename = os.path.basename(file1_path)
    f2_basename = os.path.basename(file2_path)

    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            line_num = 0
            while True:
                line_num += 1
                line1 = f1.readline()
                line2 = f2.readline()

                # --- Check for end of files ---
                if not line1 and not line2:
                    # Both files ended at the same time
                    break
                elif not line1:
                    # File 1 ended, but File 2 has more lines
                    any_difference_found = True
                    length_mismatch_line = line_num # Record where the mismatch starts
                    print(f"Difference: File 1 ('{f1_basename}') ends prematurely at line {line_num - 1}.")
                    print(f"           File 2 ('{f2_basename}') has additional lines starting from line {line_num}.")
                    break # Stop comparison
                elif not line2:
                    # File 2 ended, but File 1 has more lines
                    any_difference_found = True
                    length_mismatch_line = line_num # Record where the mismatch starts
                    print(f"Difference: File 2 ('{f2_basename}') ends prematurely at line {line_num - 1}.")
                    print(f"           File 1 ('{f1_basename}') has additional lines starting from line {line_num}.")
                    break # Stop comparison

                # --- Compare lines if both have content ---
                # Strip leading/trailing whitespace for comparison
                stripped_line1 = line1.strip()
                stripped_line2 = line2.strip()

                if stripped_line1 != stripped_line2:
                    if not any_difference_found: # Print header only once
                         print("Differences found on the following lines:")
                    any_difference_found = True
                    print(f"  - Line {line_num}")
                    # Don't need to compare tokens anymore, just note the line difference

            # --- End of file comparison ---
            if not any_difference_found:
                print(f"Files '{f1_basename}' and '{f2_basename}' are identical.")
                return True
            else:
                # A summary message isn't strictly necessary as details were printed above
                # print("\nComparison finished. Differences reported above.")
                return False

    except FileNotFoundError as e:
        print(f"Error: Cannot open file - {e}", file=sys.stderr)
        return False # Indicate failure due to file access
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False # Indicate failure due to other error

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <file1_path> <file2_path>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print(f"Comparing '{file1}' and '{file2}'...")
    files_are_identical = compare_files_print_diff_lines(file1, file2)

    if files_are_identical:
        sys.exit(0) # Exit successfully
    else:
        sys.exit(1) # Exit with an error code to indicate differences found