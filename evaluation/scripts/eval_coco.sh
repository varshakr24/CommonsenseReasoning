source ~/anaconda3/etc/profile.d/conda.sh
conda activate coco_score
TYPE=test
INPUT_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.src_alpha.txt
TRUTH_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.tgt.txt
PRED_FILE=~/CommonsenseReasoning/methods/unilm_based/decoded_sentences/${TYPE}

cd ~/CommonsenseReasoning/evaluation/Traditional/eval_metrics
python2 eval.py --key_file ${INPUT_FILE} --gts_file ${TRUTH_FILE} --res_file ${PRED_FILE} >> ~/results.txt
