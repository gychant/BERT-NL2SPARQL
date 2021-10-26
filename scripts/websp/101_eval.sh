PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python train_ie4sparql_rnn.py --data_dir WebQSP_0824_67 \
            --model_dir experiments_websp/101 \
            --epoch_num 1000 \
            --batch_size 16 \
            --cell_type gru \
            --max_len 300 \
            --hidden_size 128 \
            --num_layers 2 \
            --word_embed_dim 300 \
            --type_embed_dim 50 \
            --learning_rate 1e-4