DATA_FOLDER=../../data
OUTPUT_FOLDER=../../outputs/provenance_mpi

mkdir -p $OUTPUT_FOLDER

(python --version && pip list && pip freeze) > $OUTPUT_FOLDER/environment.txt

mpirun -np 2 python psd_by_trial_type.py $DATA_FOLDER/i140703-001.nix $DATA_FOLDER/l101210-001.nix --output_path=$OUTPUT_FOLDER --skip_channels="0;2,4" #> $OUTPUT_FOLDER/psd_by_trial_type.out 2>&1
