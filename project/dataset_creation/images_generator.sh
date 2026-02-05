#!/bin/bash

: '
    This script generates groups of synthetic star-field images for a dataset.

    Naming scheme:
        AAA0  AAA1  AAA2  AAA3  AAA4  AAA5  AAA6
        AAB0  AAB1  AAB2  AAB3  AAB4  AAB5  AAB6
        ...

    • The 3-letter prefix (AAA, AAB, AAC, ...) identifies a unique sky 
      position defined by RA, DE, and ROLL.

    • The trailing digit encodes the noise level:
          0 → perfect (noise-free) image
          1 → atmosphere only
          2 → low noise
          3 → higher noise
          4 → dark current
          5 → blur only
          6 → blur + dark current

    For each image, centroid data is saved in:
        dataset/centroid/centroid_<IMAGE_ID>.csv
'

# =======================
# Paths and directories
# =======================

BASE_DIR="$(pwd)/Star_images/dataset"
CENTROID_DIR="$BASE_DIR/centroid"
mkdir -p "$BASE_DIR"
mkdir -p "$CENTROID_DIR"

LOGFILE="$BASE_DIR/star_detected.txt"
> "$LOGFILE"

LOST_DIR="../lost"
PYTHON_BIN=/usr/bin/python3

# =======================
# Initial configuration
# =======================

CURRENT_ID="AAA"
N=50
BATCH_ID=0

if [ ! -x "$LOST_DIR/lost" ]; then
    echo "ERROR: LOST binary not found or not executable"
    exit 1
fi

# =======================
# Main loop
# =======================

for i in $(seq 1 $N); do

    echo "=== Generating sky position ID $CURRENT_ID ==="

    # =======================
    # ID1 — atmosphere only (controllo preliminare)
    # =======================

    CENT1="$CENTROID_DIR/centroid_${CURRENT_ID}1.csv"
    RAW1="$BASE_DIR/${CURRENT_ID}1.png"

    MAX_TRIES=50
    TRY=0
    VALID_STARS=0
    MIN_STARS=3
    MIN_SIZE=45 

    while (( VALID_STARS < MIN_STARS && TRY < MAX_TRIES )); do
        ((TRY++))

        # 1. Genera RA/DE/ROLL
        read RA DE ROLL <<< $(python3 increment_id.py angles)

        RAW0="$BASE_DIR/${CURRENT_ID}0.png"
        CENT0="$CENTROID_DIR/centroid_${CURRENT_ID}0.csv"

        pushd "$LOST_DIR" > /dev/null

        ./lost pipeline \
            --generate 1 \
            --generate-x-resolution 1024 \
            --generate-y-resolution 1024 \
            --fov 30 \
            --generate-spread-stddev 1.0 \
            --generate-read-noise-stddev 0.0 \
            --generate-dark-current 0.1 \
            --generate-shot-noise false \
            --generate-blur-de 0 \
            --generate-blur-ra 0 \
            --generate-blur-roll 0 \
            --generate-exposure 10 \
            --generate-false-stars 0 \
            --generate-perturb-centroids 0 \
            --generate-ra "$RA" \
            --generate-de "$DE" \
            --generate-roll "$ROLL" \
            --plot-raw-input "$RAW0" \
            --print-input-centroids "$CENT0"

        #popd > /dev/null

        # 2. Analizza stelle visibili
        VALID_STARS=$($PYTHON_BIN detect_star.py "$RAW0" $MIN_SIZE)
        echo "Detected $VALID_STARS stars with size >= $MIN_SIZE"
    done

    if (( VALID_STARS < MIN_STARS )); then
        echo "WARNING: Could not generate a valid sky with enough visible stars for $CURRENT_ID, skipping."
        continue
    fi
    echo "RA=$RA DE=$DE ROLL=$ROLL"

    # =======================
    # Analyze perfect image
    # =======================

    STARCOUNT=$($PYTHON_BIN ./perfect_image_algorithm.py "$RAW0")
    echo "Detected $STARCOUNT stars"

    pushd "$LOST_DIR" > /dev/null

    # =======================
    # ID1 — atmosphere only 
    # =======================

    CENT1="$CENTROID_DIR/centroid_${CURRENT_ID}1.csv"

    ./lost pipeline \
            --generate 1 \
            --generate-x-resolution 1024 \
            --generate-y-resolution 1024 \
            --fov 30 \
            --generate-spread-stddev 1.0 \
            --generate-read-noise-stddev 0.0 \
            --generate-dark-current 0.1 \
            --generate-blur-de 0 \
            --generate-blur-ra 0 \
            --generate-blur-roll 0 \
            --generate-ra "$RA" \
            --generate-de "$DE" \
            --generate-roll "$ROLL" \
            --plot-raw-input "$RAW1" \
            --print-input-centroids "$CENT1"

    # =======================
    # ID2 — low noise
    # =======================

    CENT2="$CENTROID_DIR/centroid_${CURRENT_ID}2.csv"

    ../lost/lost pipeline \
        --generate 1 \
        --generate-x-resolution 1024 \
        --generate-y-resolution 1024 \
        --fov 30 \
        --generate-spread-stddev 1 \
        --generate-read-noise-stddev 0.05 \
        --generate-dark-current 0.1 \
        --generate-blur-de 0 \
        --generate-blur-ra 0 \
        --generate-blur-roll 0 \
        --generate-ra "$RA" \
        --generate-de "$DE" \
        --generate-roll "$ROLL" \
        --plot-raw-input "$BASE_DIR/${CURRENT_ID}2.png" \
        --print-input-centroids "$CENT2"

    # =======================
    # ID3 — higher noise
    # =======================

    CENT3="$CENTROID_DIR/centroid_${CURRENT_ID}3.csv"

    ../lost/lost pipeline \
        --generate 1 \
        --generate-x-resolution 1024 \
        --generate-y-resolution 1024 \
        --fov 30 \
        --generate-spread-stddev 1 \
        --generate-read-noise-stddev 0.1 \
        --generate-dark-current 0.1 \
        --generate-blur-de 0 \
        --generate-blur-ra 0 \
        --generate-blur-roll 0 \
        --generate-ra "$RA" \
        --generate-de "$DE" \
        --generate-roll "$ROLL" \
        --plot-raw-input "$BASE_DIR/${CURRENT_ID}3.png" \
        --print-input-centroids "$CENT3"

    # =======================
    # ID4 — dark current
    # =======================

    CENT4="$CENTROID_DIR/centroid_${CURRENT_ID}4.csv"

    ../lost/lost pipeline \
        --generate 1 \
        --generate-x-resolution 1024 \
        --generate-y-resolution 1024 \
        --fov 30 \
        --generate-spread-stddev 1.0 \
        --generate-read-noise-stddev 0.05 \
        --generate-dark-current 0.4 \
        --generate-blur-de 0 \
        --generate-blur-ra 0 \
        --generate-blur-roll 0 \
        --generate-ra "$RA" \
        --generate-de "$DE" \
        --generate-roll "$ROLL" \
        --plot-raw-input "$BASE_DIR/${CURRENT_ID}4.png" \
        --print-input-centroids "$CENT4"

    # =======================
    # ID5 — blur only
    # =======================

    CENT5="$CENTROID_DIR/centroid_${CURRENT_ID}5.csv"

    ../lost/lost pipeline \
        --generate 1 \
        --generate-x-resolution 1024 \
        --generate-y-resolution 1024 \
        --fov 30 \
        --generate-spread-stddev 1.0 \
        --generate-read-noise-stddev 0.05 \
        --generate-dark-current 0.0 \
        --generate-blur-de 1.5 \
        --generate-blur-ra 1.5 \
        --generate-blur-roll 1.5 \
        --generate-ra "$RA" \
        --generate-de "$DE" \
        --generate-roll "$ROLL" \
        --plot-raw-input "$BASE_DIR/${CURRENT_ID}5.png" \
        --print-input-centroids "$CENT5"

    # =======================
    # ID6 — blur + dark current
    # =======================

    CENT6="$CENTROID_DIR/centroid_${CURRENT_ID}6.csv"

    ../lost/lost pipeline \
        --generate 1 \
        --generate-x-resolution 1024 \
        --generate-y-resolution 1024 \
        --fov 30 \
        --generate-spread-stddev 1.0 \
        --generate-read-noise-stddev 0.05 \
        --generate-dark-current 0.3 \
        --generate-blur-de 1.5 \
        --generate-blur-ra 1.5 \
        --generate-blur-roll 1.5 \
        --generate-ra "$RA" \
        --generate-de "$DE" \
        --generate-roll "$ROLL" \
        --plot-raw-input "$BASE_DIR/${CURRENT_ID}6.png" \
        --print-input-centroids "$CENT6"

    popd > /dev/null

    # =======================
    # Logging (UNCHANGED)
    # =======================

    for n in 0 1 2 3 4 5 6; do
        echo "${CURRENT_ID}${n}  $RA  $DE  $ROLL  $STARCOUNT" >> "$LOGFILE"
    done

    # =======================
    # Next ID
    # =======================

    CURRENT_ID=$(python3 ./increment_id.py increment_id "$CURRENT_ID")

    # =======================
    # Archive batches
    # =======================

    if (( i % 50 == 0 )); then
        BATCH_ID=$(printf "%03d" "$((BATCH_ID+1))")
        ARCHIVE_NAME="batch_${BATCH_ID}.tar.gz"

        tar -czf "$BASE_DIR/../data/$ARCHIVE_NAME" -C "$BASE_DIR" .
        rm -f "$BASE_DIR"/*.png
    fi

done
