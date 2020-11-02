# git clone https://github.com/varshakr24/CommonsenseReasoning.git
# cd CommonsenseReasoning/methods/unilm_based
yes | conda create -n unilm_env python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate unilm_env
yes | conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch -n unilm_env

mkdir tmp
pip install gdown
gdown https://drive.google.com/uc?id=1Zj_nZWO7YffaOInj3Q4SZyn09Mb3In-e
unzip unilmv1-large-cased.zip
rm unilmv1-large-cased.zip
mv unilmv1-large-cased.bin tmp/

cd tmp
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ../../


pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk 
python -c "import nltk; nltk.download('punkt')"
pip install -e git://github.com/Maluuba/nlg-eval.git#egg=nlg-eval
cd unilm/src
pip install --user --editable .
cd ../../

conda activate unilm_env; \
DATA_DIR=../../dataset/final_data/commongen/; \
OUTPUT_DIR=tmp/finetuned_models/; \
MODEL_RECOVER_PATH=tmp/unilmv1-large-cased.bin; \
export PYTORCH_PRETRAINED_BERT_CACHE=tmp/bert-cased-pretrained-cache; \
CUDA_VISIBLE_DEVICES=0 python unilm/src/biunilm/run_seq2seq.py --do_train --num_workers 0 \
  --bert_model bert-large-cased --new_segment_ids \
  --data_dir ${DATA_DIR} \
  --src_file commongen.train.src_alpha.txt \
  --tgt_file commongen.train.tgt.txt \
  --cs_file commongen.train.cs_rel.txt \
  --exp_file commongen.train.exp.txt \
  --output_dir ${OUTPUT_DIR}/bert_save \
  --log_dir ${OUTPUT_DIR}/bert_log \
  --model_recover_path ${MODEL_RECOVER_PATH} \
  --max_seq_length 64 --max_position_embeddings 64 \
  --always_truncate_tail  --max_len_a 64 --max_len_b 64 \
  --mask_prob 0.7 --max_pred 20 \
  --train_batch_size 32 --gradient_accumulation_steps 1 \
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 \
  --num_train_epochs 10 \
  # --fp16 --amp  \
