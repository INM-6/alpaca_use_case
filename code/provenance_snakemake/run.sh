OUTPUT_FOLDER=../../../outputs/provenance_snakemake

(
  cd workflow || exit

  mkdir -p $OUTPUT_FOLDER

  (python --version && pip list && pip freeze) > $OUTPUT_FOLDER/environment.txt

  snakemake $1 --cores 2 > $OUTPUT_FOLDER/snakemake.out 2>&1

  python ../generate_gexf_from_prov.py --source_path=$OUTPUT_FOLDER --output=$OUTPUT_FOLDER/R2G_PSD_all_subjects.gexf
)