# export HF_DATASETS_CACHE= SET IT HERE
# export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64


# HNMR
python ./benchmark/run_cnn_jung_baseline.py \
--analytical_data ./data/ \
--out_path ./runs_cnn_f_groups/hnmr \
--column h_nmr_spectra

# CNMR
python ./benchmark/run_cnn_jung_baseline.py \
--analytical_data ./data/ \
--out_path ./runs_cnn_f_groups/cnmr \
--column c_nmr_spectra

# IR
python ./benchmark/run_cnn_jung_baseline.py \
--analytical_data ./data/ \
--out_path ./runs_cnn_f_groups/ir \
--column ir_spectra

# Pos MSMS
python ./benchmark/run_cnn_jung_baseline.py \
--analytical_data ./data/ \
--out_path ./runs_cnn_f_groups/pos_msms \
--column pos_msms

# IR
python ./benchmark/run_cnn_jung_baseline.py \
--analytical_data ./data/ \
--out_path ./runs_cnn_f_groups/neg_msms \
--column neg_msms
