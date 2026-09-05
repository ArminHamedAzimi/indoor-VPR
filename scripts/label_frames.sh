#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Label an inclusive range of numbered image frames and save the labels as CSV.

Usage:
  ./scripts/label_frames.sh [--overwrite] IMAGE_DIR CSV_FILE START END LABEL
  ./scripts/label_frames.sh

Example:
  ./scripts/label_frames.sh \
    datasets/dataset-2/frames-IMG_3610 \
    datasets/dataset-2/labels/frames-IMG_3610.csv \
    1 250 corridor_a

Options:
  --overwrite  Replace existing rows for frames in the selected range.
  -h, --help   Show this help message.

With no arguments, the script asks for each value interactively.
EOF
}

overwrite=false
if [[ ${1:-} == "--overwrite" ]]; then
    overwrite=true
    shift
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if (( $# == 0 )); then
    read -r -p "Image directory: " image_dir
    read -r -p "Output CSV file: " csv_file
    read -r -p "First frame number: " start_frame
    read -r -p "Last frame number (inclusive): " end_frame
    read -r -p "Label: " label
elif (( $# == 5 )); then
    image_dir=$1
    csv_file=$2
    start_frame=$3
    end_frame=$4
    label=$5
else
    usage >&2
    exit 2
fi

if [[ ! -d $image_dir ]]; then
    echo "Error: image directory does not exist: $image_dir" >&2
    exit 1
fi

if [[ ! $start_frame =~ ^[0-9]+$ || ! $end_frame =~ ^[0-9]+$ ]]; then
    echo "Error: START and END must be non-negative frame numbers." >&2
    exit 1
fi

start_number=$((10#$start_frame))
end_number=$((10#$end_frame))
if (( start_number > end_number )); then
    echo "Error: START must be less than or equal to END." >&2
    exit 1
fi

if [[ -z $label ]]; then
    echo "Error: label cannot be empty." >&2
    exit 1
fi

if [[ $label == *$'\n'* || $label == *$'\r'* ]]; then
    echo "Error: label cannot contain a newline." >&2
    exit 1
fi

declare -a selected_paths=()
declare -a selected_indices=()
declare -a missing_paths=()

for ((index = start_number; index <= end_number; index++)); do
    printf -v filename 'frame_%06d.jpg' "$index"
    path="$image_dir/$filename"
    if [[ -f $path ]]; then
        selected_paths+=("$path")
        selected_indices+=("$index")
    else
        missing_paths+=("$path")
    fi
done

if (( ${#missing_paths[@]} > 0 )); then
    echo "Error: ${#missing_paths[@]} selected image(s) do not exist." >&2
    printf '  %s\n' "${missing_paths[@]:0:10}" >&2
    if (( ${#missing_paths[@]} > 10 )); then
        echo "  ... and $((${#missing_paths[@]} - 10)) more" >&2
    fi
    exit 1
fi

if (( ${#selected_paths[@]} == 0 )); then
    echo "Error: the selected range contains no images." >&2
    exit 1
fi

mkdir -p "$(dirname "$csv_file")"

if [[ -f $csv_file && $overwrite == false ]]; then
    duplicate_count=$(awk -F, -v first="$start_number" -v last="$end_number" \
        'NR > 1 && ($2 + 0) >= first && ($2 + 0) <= last { count++ } END { print count + 0 }' \
        "$csv_file")
    if (( duplicate_count > 0 )); then
        echo "Error: $duplicate_count frame(s) in this range are already labeled." >&2
        echo "Run again with --overwrite to replace their labels." >&2
        exit 1
    fi
fi

echo "Frames: $start_number through $end_number (${#selected_paths[@]} images)"
echo "Label:  $label"
echo "CSV:    $csv_file"
read -r -p "Write these labels? [y/N] " confirmation
if [[ ! $confirmation =~ ^[Yy]$ ]]; then
    echo "Cancelled; no labels were changed."
    exit 0
fi

temporary_csv=$(mktemp "${TMPDIR:-/tmp}/indoor-vpr-labels.XXXXXX")
trap 'rm -f "$temporary_csv"' EXIT

if [[ -f $csv_file ]]; then
    if [[ $overwrite == true ]]; then
        awk -F, -v first="$start_number" -v last="$end_number" \
            'NR == 1 || !(($2 + 0) >= first && ($2 + 0) <= last)' \
            "$csv_file" > "$temporary_csv"
    else
        cp "$csv_file" "$temporary_csv"
    fi
else
    printf 'image_path,frame_index,label\n' > "$temporary_csv"
fi

escaped_label=${label//\"/\"\"}
for array_index in "${!selected_paths[@]}"; do
    escaped_path=${selected_paths[$array_index]//\"/\"\"}
    printf '"%s",%d,"%s"\n' \
        "$escaped_path" "${selected_indices[$array_index]}" "$escaped_label" \
        >> "$temporary_csv"
done

mv "$temporary_csv" "$csv_file"
trap - EXIT

echo "Saved ${#selected_paths[@]} labels to $csv_file"
