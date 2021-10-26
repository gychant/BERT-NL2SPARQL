PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python train_ie4sparql_rnn.py --data_dir lic-corrected-70-semantic-embedd-v4 \
            --model_dir experiments_lcquad/201 \
            --epoch_num 1000 \
            --batch_size 16 \
            --cell_type gru \
            --max_len 300 \
            --hidden_size 300 \
            --num_layers 2 \
            --word_embed_dim 300 \
            --type_embed_dim 128 \
            --learning_rate 5e-4 \
            --dropout 0.2 \
            --dynamic_threshold
            