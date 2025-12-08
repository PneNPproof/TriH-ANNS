#!/bin/bash

# --- Configuration Section ---

# Specify the path to your application
APP_NAME="./trih/bin/trih_anns" # Change this to the actual path if needed

# Specify the data files to test
# DATA_FILES=("gist-960-euclidean-shuffle.hdf5" "cohere-768-euclidean-shuffle.hdf5" "openai-1536-euclidean-shuffle.hdf5" "sift-128-euclidean-shuffle.hdf5" "msong-420-euclidean-shuffle.hdf5" "imagenet-150-euclidean-shuffle.hdf5") # Add more data filenames here
DATA_FILES=("self-v3-2048-euclidean.hdf5")
# Map data filenames to their corresponding index tag prefixes
declare -A DATA_TO_TAG_PREFIX=(
    ["laion-512-euclidean-shuffle.hdf5"]="laion"
    ["gist-960-euclidean-shuffle.hdf5"]="gist"
    ["cohere-768-euclidean-shuffle.hdf5"]="cohere"
    ["openai-1536-euclidean-shuffle.hdf5"]="openai"
    ["sift-128-euclidean-shuffle.hdf5"]="sift"
    ["msong-420-euclidean-shuffle.hdf5"]="msong"
    ["imagenet-150-euclidean-shuffle.hdf5"]="imagenet"
    ["fashion-mnist-784-euclidean-divide-256.hdf5"]="mnist" # Example for fashion-mnist
    ["self-v3-2048-euclidean.hdf5"]="custom"
    # Add mappings for other data files here
)

# Map data filenames to their respective column_num values (space-separated)
declare -A DATA_TO_COLUMNS=(
    ["gist-960-euclidean-shuffle.hdf5"]="64 128 256 384"         
    ["cohere-768-euclidean-shuffle.hdf5"]="64 128 256 384"       
    ["openai-1536-euclidean-shuffle.hdf5"]="64 128 256 384"      
    ["sift-128-euclidean-shuffle.hdf5"]="32 64"                 
    ["msong-420-euclidean-shuffle.hdf5"]="64 128"                
    ["imagenet-150-euclidean-shuffle.hdf5"]="32 64"
    ["laion-512-euclidean-shuffle.hdf5"]="256"
    ["fashion-mnist-784-euclidean-divide-256.hdf5"]="256 392"              
    ["self-v3-2048-euclidean.hdf5"]="192 256 384"
    # Add mappings for other data files and their columns here
    # Example with multiple columns: ["mydata.bin"]="64 128 256"
)

# Fixed ratio for index construction
RATIO=0.9

# --- Search Phase Parameters ---

# Possible values for search parameters (space-separated)
QUERY_BATCH_SIZES=(16 1000)
PHASE1_TOPKS=(100 110 120 130 140 150)
REDUCE_GROUP_SIZES=(64 128 256)

# Fixed search parameters
PHASE2_TOPK=100
NUM_WORKERS=4
BATCH_NUM_PER_WORKER=192
RERANK_THREAD_POOL_SIZE=20

# Directory to store results (optional, as we now use one file)
# RESULTS_DIR="anns_results" # No longer strictly needed for separate files

# --- Results Directory and File ---
RESULTS_DIR="anns_results" # Fixed directory name
TIMESTAMP=$(date '+%Y%m%d_%H%M%S') # Timestamp for the filename
SEARCH_RESULTS_BASE_NAME="search_summary" # Base name for the summary file
SEARCH_RESULTS_EXT=".txt" # Or .csv if you prefer

# --- Helper Functions ---

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# --- Create Results Directory (ensure the fixed one exists) ---
# This section should be placed after Sanity Checks but before initializing the file
mkdir -p "$RESULTS_DIR"
if [ $? -ne 0 ]; then
    log "ERROR: Failed to create results directory '$RESULTS_DIR'."
    exit 1
fi
log "Results will be stored in the directory '$RESULTS_DIR'" # Adjusted log message

# --- Construct Full Path for Search Results File ---
# Combines fixed dir, base name, timestamp, and extension
SEARCH_RESULTS_FILE="${RESULTS_DIR}/${SEARCH_RESULTS_BASE_NAME}_${TIMESTAMP}${SEARCH_RESULTS_EXT}"

# Clear or initialize the single results file (using the new dynamic path)
echo "# ANNS Search Results - Run started at $(date)" > "$SEARCH_RESULTS_FILE"
echo "# Grouped by (index_tag, query_batch_size, reduce_group_size)" >> "$SEARCH_RESULTS_FILE"
echo "" >> "$SEARCH_RESULTS_FILE" # Add an empty line
log "Search results will be appended to '$SEARCH_RESULTS_FILE'" # $SEARCH_RESULTS_FILE now holds the full dynamic path



# --- Sanity Checks ---

if [ ! -x "$APP_NAME" ]; then
    log "ERROR: Application '$APP_NAME' not found or not executable."
    exit 1
fi

if [ ${#DATA_FILES[@]} -eq 0 ]; then
    log "ERROR: No data files specified in DATA_FILES array."
    exit 1
fi

# Check if bash version supports associative arrays (Bash 4.0+)
if [[ -z "${BASH_VERSINFO[0]}" || "${BASH_VERSINFO[0]}" -lt 4 ]]; then
  log "ERROR: This script requires Bash version 4.0 or later for associative arrays."
  exit 1
fi

# Create results directory (still useful for index files potentially)
# mkdir -p "$RESULTS_DIR"
# log "Intermediate files might be generated in the current directory or as defined by '$APP_NAME'" # Adjusted message




# --- Phase 1: Index Construction ---

# log "--- Starting Phase 1: Index Construction ---"

# for data_file in "${DATA_FILES[@]}"; do
#     log "Processing data file: $data_file"

#     if [[ -z "${DATA_TO_TAG_PREFIX[$data_file]}" ]]; then
#         log "WARNING: No tag prefix defined for $data_file in DATA_TO_TAG_PREFIX. Skipping."
#         continue
#     fi
#     tag_prefix="${DATA_TO_TAG_PREFIX[$data_file]}"

#     if [[ -z "${DATA_TO_COLUMNS[$data_file]}" ]]; then
#         log "WARNING: No column numbers defined for $data_file in DATA_TO_COLUMNS. Skipping."
#         continue
#     fi

#     # Read column numbers into an array
#     read -ra col_num_array <<< "${DATA_TO_COLUMNS[$data_file]}"

#     for col_num in "${col_num_array[@]}"; do
#         index_tag="${tag_prefix}_${col_num}"
#         log "Constructing index: $index_tag for $data_file (column: $col_num)"

#         # Construct the command
#         build_cmd=("$APP_NAME" "$data_file" "b" "$index_tag" "$col_num" "$RATIO")

#         # log "Executing: ${build_cmd[@]}"
#         log "Executing: ${build_cmd[*]}"
#         # Execute the command
#         if ! "${build_cmd[@]}"; then
#             log "ERROR: Index construction failed for $index_tag ($data_file, col $col_num)."
#             # Decide if you want to exit or continue with other constructions/searches
#             # exit 1 # Uncomment to exit on failure
#         else
#             log "Index $index_tag construction completed successfully."
#         fi
#         echo # Add a newline for better readability
#     done
# done

# log "--- Finished Phase 1: Index Construction ---"
# echo

# --- Phase 2: Search Testing ---

log "--- Starting Phase 2: Search Testing ---"

for data_file in "${DATA_FILES[@]}"; do
    log "Testing searches for data file: $data_file"

    if [[ -z "${DATA_TO_TAG_PREFIX[$data_file]}" ]]; then
        log "WARNING: No tag prefix defined for $data_file. Skipping search tests."
        continue
    fi
    tag_prefix="${DATA_TO_TAG_PREFIX[$data_file]}"

    if [[ -z "${DATA_TO_COLUMNS[$data_file]}" ]]; then
        log "WARNING: No column numbers defined for $data_file. Skipping search tests."
        continue
    fi

    read -ra col_num_array <<< "${DATA_TO_COLUMNS[$data_file]}"

    for col_num in "${col_num_array[@]}"; do
        index_tag="${tag_prefix}_${col_num}"
        log "Testing index: $index_tag"

        # Optional: Check if the corresponding index file/directory exists
        # ... (add check here if needed) ...

        # --- Loop order changed here ---
        for qbs in "${QUERY_BATCH_SIZES[@]}"; do
            for rgs in "${REDUCE_GROUP_SIZES[@]}"; do

                # --- Group Header ---
                # Write the header for this specific group (index_tag, qbs, rgs)
                # to the single output file *before* iterating through phase1_topks.
                log "--- Preparing group: index=$index_tag qbs=$qbs rgs=$rgs ---"
                echo "(index_tag=${index_tag}, query_batch_size=${qbs}, reduce_group_size=${rgs})" >> "$SEARCH_RESULTS_FILE"
                # Optionally add a CSV-like header for the data lines that follow
                echo "phase1_topk,qps,recall@100" >> "$SEARCH_RESULTS_FILE" # Header for the results in this block

                # --- Innermost Loop: Iterate through phase1_topk for this group ---
                for p1tk in "${PHASE1_TOPKS[@]}"; do

                    # Construct the search command
                    search_cmd=(
                        "$APP_NAME"
                        "$data_file"
                        "s"
                        "$index_tag"
                        "$qbs"
                        "$p1tk"
                        "$PHASE2_TOPK"
                        "$rgs"
                        "$NUM_WORKERS"
                        "$BATCH_NUM_PER_WORKER"
                        "$RERANK_THREAD_POOL_SIZE"
                    )

                    log "Executing search: p1tk=$p1tk (Group: index=$index_tag qbs=$qbs rgs=$rgs)"
                    # log "Command: ${search_cmd[@]}" # You might want to comment this out if it gets too verbose
                    log "Command: ${search_cmd[*]}"

                    # Execute and capture output
                    output=$( "${search_cmd[@]}" )
                    exit_status=$?

                    if [ $exit_status -ne 0 ]; then
                        log "ERROR: Search command failed with exit status $exit_status for p1tk=$p1tk in group (index=$index_tag qbs=$qbs rgs=$rgs)."
                        log "Failed command: ${search_cmd[@]}"
                        # Append failure indication to the results file
                        echo "${p1tk},FAILED,FAILED" >> "$SEARCH_RESULTS_FILE"
                        continue # Continue to the next p1tk
                    fi

                    # Parse the output
                    qps=$(echo "$output" | grep 'qps =' | awk '{print $3}')
                    recall=$(echo "$output" | grep 'recall@100:' | awk '{printf "%.6f", $2}')

                    # Validate parsed values
                    if [[ -z "$qps" || -z "$recall" ]]; then
                        log "ERROR: Could not parse QPS or Recall from output for p1tk=$p1tk in group (index=$index_tag qbs=$qbs rgs=$rgs)."
                        log "Output was:"
                        echo "$output"
                        # Append parse error indication to the results file
                        echo "${p1tk},PARSE_ERROR,PARSE_ERROR" >> "$SEARCH_RESULTS_FILE"
                        continue # Continue to the next p1tk
                    fi

                    # Append the result line (p1tk, qps, recall) for the current p1tk
                    # to the single results file under the current group header.
                    echo "${p1tk},${qps},${recall}" >> "$SEARCH_RESULTS_FILE"
                    log "Result: p1tk=$p1tk QPS=$qps Recall=$recall -> Appended to $SEARCH_RESULTS_FILE"

                done # End phase1_topk loop

                # Add a blank line after each group's results for readability
                echo "" >> "$SEARCH_RESULTS_FILE"
                log "--- Finished group: index=$index_tag qbs=$qbs rgs=$rgs ---"
                echo # Add a newline to console log

            done # End reduce_group_size loop
        done # End query_batch_size loop
    done # End col_num loop
done # End data_file loop

log "--- Finished Phase 2: Search Testing ---"
# log "All tests completed. Results are appended to '$SEARCH_RESULTS_FILE'."
log "All tests completed. Search results summary saved as '$SEARCH_RESULTS_FILE' in the '$RESULTS_DIR' directory."