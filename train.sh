# training schedular
conda env: py39
conda activate py39

pip3 install torch --index-url https://download.pytorch.org/whl/cu118

export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

pip install numpy==1.26.0
pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# Training schedular
# File 1 -> GPU 2, cell_line:HEK293FT, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='cell_line:HEK293FT' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, match:2, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:2' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: seq, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: seq_bert, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: no_mfe, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_icshape_binding_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: no_icshape, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_binding_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: no_binding, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_icshape_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, model: no_binding & icshape, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 1 -> GPU 2, add7, DONE
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_fold_mfe1_mfe2' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3

# File 2 -> GPU 3, match:6, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:6' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, match:4, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:4' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, match:1, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:1' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, random, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add1, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add2, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add3, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, only relatelen_utrrate, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, model: no relatelen_utrrate, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add_single_celline:HEK293FT, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py add_single_celline --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='cell_line:HEK293FT:single' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, abcc, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='abcc' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, full_icshape1, Done
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/home/postphd/SJD/cas13d_data/icshape_filtered_data/full-icshape-new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
# File 2 -> GPU 3, full_icshape1, Seq, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/home/postphd/SJD/cas13d_data/icshape_filtered_data/full-icshape-new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'



# File 3 -> GPU 4, match:5, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:5' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, match:3, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:3' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, match:0, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='match:0' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, only mfe, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_mfe1_mfe2' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, only icshape, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_icshape' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, only binding, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_binding' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, only relatelen_utrrate, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, model: no bert, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_fold_mfe1_mfe2_icshape_binding' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, full_icshape1, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/home/postphd/SJD/cas13d_data/icshape_filtered_data/icshape_filtered_new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'
# File 3 -> GPU 4, full_icshape1, seq, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq' --data_des='random' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/home/postphd/SJD/cas13d_data/icshape_filtered_data/icshape_filtered_new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'



# File 2 -> GPU 3, add_single_celline:K562, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py add_single_celline --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='cell_line:K562:single' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add_single_celline:A375, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py add_single_celline --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='cell_line:A375:single' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 2 -> GPU 3, add_single_celline:HAP1, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py add_single_celline --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='cell_line:HAP1:single' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, abca, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='abca' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# File 3 -> GPU 4, abcb, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='abcb' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3
# random_all, File 3, pure training, DONE
python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 1, pure training, DONE
python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'


# random_all, File 1, pure training cell line:K562, 
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:K562' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 2, pure training cell line:A375, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:A375' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 3, pure training cell line:HAP1, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 2, pure training cell line:HEK293FT, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HEK293FT' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'

# seq
# random_all, File 1, pure training cell line:K562, 
cd /data3/SJD/Ca13TransformerDesigned1 && python3 -u logicArchi.py control --gpu_ids=[2] --model_des='seq' --data_des='random_all:K562' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 2, pure training cell line:A375, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq' --data_des='random_all:A375' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 3, pure training cell line:HAP1, DONE
cd /data3/SJD/Ca13TransformerDesigned3 && python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
# random_all, File 2, pure training cell line:HEK293FT, DONE
cd /data3/SJD/Ca13TransformerDesigned2 && python3 -u logicArchi.py control --gpu_ids=[3] --model_des='seq' --data_des='random_all:HEK293FT' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3  --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'




# model 40
# random_all, File 3, pure training, DONE
python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'

# data 300
python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader300' --lr=1e-3 --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random.csv'
python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader300' --lr=1e-3 --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/train_random_train_random.csv'

# model 4
python3 -u logicArchi.py control --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --lr=1e-3 --data_path='/data3/SJD/Ca13TransformerDesigned3/shape_data/ours/train_random:full_icshape.csv'



###
Jindong Sun, jdsun@sdust.edu.cn, Shandong University of Science and Technology, College of Intelligent Equipment, Qingdao, 266590
Jindong Sun, jdsun@sdust.edu.cn, Shandong University of Science and Technology, College of Intelligent Equipment, Taian, 271000

# val
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_random.csv'


# python3 -u logicArchi.py val --gpu_ids=[3] --model_des='seq' --data_des='random_all:HEK293FT' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/home/postphd/SJD/cas13d_data/final_data/new-all_merge_remove_repeat_containmismatch_rpkm_cds_utrinfo_depth.csv'

# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HEK293FT.csv'

# random_all
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:A375.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:K562.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HAP1.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HEK293FT.csv'

# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:A375.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:K562.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HAP1.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq_bert_fold_mfe1_mfe2_icshape_binding_relatelen_utrrate' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HEK293FT.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:A375.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:K562.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HAP1.csv'
# python3 -u logicArchi.py val --gpu_ids=[4] --model_des='seq' --data_des='random_all:HAP1' --model='SCALPEL' --dataset='BertOnehotLoader30' --data_path='/data3/SJD/Ca13TransformerDesigned3/other_model_data/ours/val_vcell:HEK293FT.csv'

