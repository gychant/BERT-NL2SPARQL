PYTHONIOENCODING=utf-8
export PYTHONIOENCODING

python train_ie4sparql_bert.py --data_dir data/lcquad/latest \
            --model_dir experiments_lcquad/bert/202 \
            --bert_model_dir uncased_L-24_H-1024_A-16 \
            --merge_predicates \
            --do_eval
