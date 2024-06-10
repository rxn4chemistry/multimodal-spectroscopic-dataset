# H NMR
python ./benchmark/run_xgb_baseline.py \
    --analytical_data ./data\
    --out_path ./runs_xgb_f_groups/h_nmr \
    --column h_nmr_spectra

# C NMR
python ./benchmark/run_xgb_baseline.py \
    --analytical_data ./data\
    --out_path ./runs_xgb_f_groups/c_nmr \
    --column c_nmr_spectra

# IR
python ./benchmark/run_xgb_baseline.py \
    --analytical_data ./data\
    --out_path ./runs_xgb_f_groups/ir \
    --column ir_spectra

# Pos MSMS
python ./benchmark/run_xgb_baseline.py \
    --analytical_data ./data\
    --out_path ./runs_xgb_f_groups/pos_msms \
    --column pos_msms

# Neg MSMS
python ./benchmark/run_xgb_baseline.py \
    --analytical_data ./data\
    --out_path ./runs_xgb_f_groups/neg_msms \
    --column neg_msms
