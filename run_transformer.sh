#!/bin/bash

export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

run_dir=./runs

# Run replication for structure prediction from spectra

# H NMR
python ./benchmark/generate_input.py \
        --analytical_data ./data/ \
        --out_path ${run_dir}/runs_new_onmt_w_formula/h_nmr \
        --formula \
        --h_nmr 

python ./benchmark/start_training.py \
        --out_path ${run_dir}/runs_new_onmt_w_formula/h_nmr 
 

# C NMR
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_new_onmt_w_formula/c_nmr \
        --formula \
        --c_nmr

python ./benchmark/start_training.py \
        --out_path ${run_dir}/runs_new_onmt_w_formula/h_nmr 


# C NMR + H NMR
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_new_onmt_w_formula/c_nmr \
        --formula \
        --h_nmr \
        --c_nmr

python ./benchmark/start_training.py \
        --out_path ${run_dir}/runs_new_onmt_w_formula/h_nmr 

