import sys
import os

def compare_topk_files_all_diffs(file1_path, file2_path):
    """
    Compares two text files containing space-separated IDs line by line.

    Reports ALL differences found, including line number and token position.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.

    Returns:
        bool: True if files are identical, False otherwise.
              Prints all difference details to stdout if not identical.
              Prints error details to stderr if files cannot be opened.
    """
    any_difference_found = False
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
                    # Both files ended at the same time, break the loop
                    break
                elif not line1:
                    # File 1 ended, but File 2 has more lines
                    if not any_difference_found: # Print header only once for length diff
                         print("-" * 20) # Separator
                    print(f"Difference: File 1 ('{f1_basename}') ended prematurely.")
                    print(f"  File 2 ('{f2_basename}') has extra content starting at line {line_num}.")
                    print(f"  Line {line_num} in File 2: '{line2.strip()}'")
                    any_difference_found = True
                    # Continue reading and printing remaining lines from file 2
                    while True:
                        line2 = f2.readline()
                        if not line2:
                            break
                        line_num += 1
                        print(f"  Extra Line {line_num} in File 2: '{line2.strip()}'")
                    break # Reached end of file 2, exit main loop
                elif not line2:
                    # File 2 ended, but File 1 has more lines
                    if not any_difference_found: # Print header only once for length diff
                         print("-" * 20) # Separator
                    print(f"Difference: File 2 ('{f2_basename}') ended prematurely.")
                    print(f"  File 1 ('{f1_basename}') has extra content starting at line {line_num}.")
                    print(f"  Line {line_num} in File 1: '{line1.strip()}'")
                    any_difference_found = True
                    # Continue reading and printing remaining lines from file 1
                    while True:
                        line1 = f1.readline()
                        if not line1:
                            break
                        line_num += 1
                        print(f"  Extra Line {line_num} in File 1: '{line1.strip()}'")
                    break # Reached end of file 1, exit main loop

                # --- Compare lines if both have content ---
                stripped_line1 = line1.strip()
                stripped_line2 = line2.strip()

                if stripped_line1 != stripped_line2:
                    # Lines differ, find and report all differing tokens on this line
                    if not any_difference_found:
                         print("-" * 20) # Separator before first reported difference
                    any_difference_found = True # Mark that at least one difference exists
                    print(f"Difference found on Line {line_num}:")
                    print(f"  (Full Line File 1: '{stripped_line1}')")
                    print(f"  (Full Line File 2: '{stripped_line2}')")

                    tokens1 = stripped_line1.split()
                    tokens2 = stripped_line2.split()
                    max_len = max(len(tokens1), len(tokens2))

                    for i in range(max_len):
                        token1 = tokens1[i] if i < len(tokens1) else "<MISSING>"
                        token2 = tokens2[i] if i < len(tokens2) else "<MISSING>"

                        if token1 != token2:
                            print(f"  - Position {i + 1}: File 1='{token1}', File 2='{token2}'")
                    print("-" * 20) # Separator after each differing line's details

            # --- End of file comparison ---
            if not any_difference_found:
                print(f"Files '{f1_basename}' and '{f2_basename}' are identical.")
                return True
            else:
                print("\nComparison finished. Differences listed above.")
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
    files_are_identical = compare_topk_files_all_diffs(file1, file2)

    if files_are_identical:
        sys.exit(0) # Exit successfully
    else:
        sys.exit(1) # Exit with an error code to indicate differences found