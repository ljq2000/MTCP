python train.py \
        --label_json='../dataset/SHA/SHA_train_label_20%.json' \
        --unlabel_json='../dataset/SHA/SHA_train_unlabel_20%.json' \
        --test_json='../dataset/SHA/SHA_test.json' \
        --gpu='2'\
        --data_name='SHA' \
        --per_label='20%'\
        --task_id='sha_20'\
        --net_name='csrnet'\
        --optimizer='adam' \
        --lr=1e-6\
        --lr_epoch=30\
        --lr_decay_rate=0.5\
        --epochs=200\
        --alpha=0.2\
        --weight_decay=5e-4\
        --seed=-1\
        --tln_mul=4\
        --resume=''\
        --pre=''\

