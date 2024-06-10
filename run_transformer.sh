#!/bin/bash

export LD_LIBRARY_PATH=/opt/share/gcc-10.1.0//lib64:/opt/share/gcc-10.1.0//lib:/usr/local/cuda-12.2/lib64

run_dir=./runs

# H NMR
python ./benchmark/generate_input.py \
        --analytical_data ./data/ \
        --out_path ${run_dir}/runs_f_groups/h_nmr \
        --h_nmr 

python ./benchmark/start_training.py \
        --output_path ${run_dir}/runs_f_groups/h_nmr \
        --template_path ./benchmark/transformer_template.yaml
# C NMR
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_f_groups/c_nmr \
        --c_nmr

python ./benchmark/start_training.py\
            --output_path ${run_dir}/runs_f_groups/c_nmr \
             --template_path ./benchmark/transformer_template.yaml

# IR
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path {$run_dir}/runs_f_groups/ir \
        --ir

python ./benchmark/start_training.py\
            --output_path ${run_dir}/runs_f_groups/ir \
             --template_path ./benchmark/transformer_template.yaml

# Pos MSMS
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_f_groups/pos_msms \
        --pos_msms

python ./benchmark/start_training.py\
            --output_path ${run_dir}/runs_f_groups/pos_msms \
             --template_path ./benchmark/transformer_template.yaml

# Neg MSMS

python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_f_groups/neg_msms \
        --neg_msms
python ./benchmark/start_training.py\
            --output_path ${run_dir}/runs_f_groups/neg_msms \
             --template_path ./benchmark/transformer_template.yaml

# 1H + 13C
python ./benchmark/generate_input.py \
        --analytical_data  ./data/ \
        --out_path ${run_dir}/runs_f_groups/all_modalities \
        --h_nmr \
        --c_nmr \

python ./benchmark/start_training.py\
            --output_path ${run_dir}/runs_f_groups/all_modalities \
             --template_path ./benchmark/transformer_template.yaml

