TYPE=test
INPUT_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.src_alpha.txt
TRUTH_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.tgt.txt
PRED_FILE=~/CommonsenseReasoning/methods/unilm_based/decoded_sentences/${TYPE}

cd ~/CommonsenseReasoning/methods/unilm_based
~/anaconda3/envs/unilm_env/bin/python unilm/src/gigaword/eval.py --pred ${PRED_FILE}   --gold ${TRUTH_FILE} --perl
