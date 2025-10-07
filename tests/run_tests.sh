#!/bin/bash

# Set default parameters
default_grid_size_lat=1
default_grid_size_lon=1
default_run_distributed=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)

            bold=$(tput bold)
            normal=$(tput sgr0)

            echo "Runs the torch-harmonics test suite."
            echo "${bold}Arguments:${normal}"
            echo "  ${bold}-h   | --help:${normal} Prints this text."
            echo "  ${bold}-d   | --run_distributed:${normal} Run the distributed test suite."
            echo "  ${bold}-lat | --grid_size_lat:${normal} Number of ranks in latitudinal direction for distributed case."
            echo "  ${bold}-lon | --grid_size_lon:${normal} Number of ranks in longitudinal direction for distributed case."

            shift
            exit 0
            ;;
        -lat|--grid_size_lat)
            grid_size_lat="$2"
            shift 2
            ;;
        -lon|--grid_size_lon)
            grid_size_lon="$2"
            shift 2
            ;;
        -d|--run_distributed)
            run_distributed=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Use default values if arguments were not provided
grid_size_lat=${grid_size_lat:-$default_grid_size_lat}
grid_size_lon=${grid_size_lon:-$default_grid_size_lon}
run_distributed=${run_distributed:-$default_run_distributed}

echo "Running sequential tests:"
python3 -m pytest tests/test_convolution.py tests/test_sht.py

# Run distributed tests if requested
if [ "$run_distributed" = "true" ]; then

    echo "Running distributed tests with the following parameters:"
    echo "Grid size latitude: $grid_size_lat"
    echo "Grid size longitude: $grid_size_lon"

    ngpu=$(( ${grid_size_lat} * ${grid_size_lon} ))

    mpirun --allow-run-as-root -np ${ngpu} bash -c "
        export CUDA_LAUNCH_BLOCKING=1;
        export WORLD_RANK=\${OMPI_COMM_WORLD_RANK};
        export WORLD_SIZE=\${OMPI_COMM_WORLD_SIZE};
        export RANK=\${OMPI_COMM_WORLD_RANK};
        export MASTER_ADDR=localhost;
        export MASTER_PORT=29501;
        export GRID_H=${grid_size_lat};
        export GRID_W=${grid_size_lon};
        python3 -m pytest tests/test_distributed_sht.py
        python3 -m pytest tests/test_distributed_convolution.py
        python3 -m pytest tests/test_distributed_attention.py
        python3 -m pytest tests/test_distributed_resample.py
        "
else
    echo "Skipping distributed tests."
fi
