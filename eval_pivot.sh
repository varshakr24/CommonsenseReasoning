TYPE=test
INPUT_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.src_alpha.txt
TRUTH_FILE=~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.tgt.txt
PRED_FILE=~/CommonsenseReasoning/methods/unilm_based/decoded_sentences/${TYPE}

cd evaluation/PivotScore
~/anaconda3/envs/pivot_score/bin/python evaluate.py --pred ${PRED_FILE}   --ref ${TRUTH_FILE} --cs ${INPUT_FILE} --cs_str ~/CommonsenseReasoning/dataset/final_data/commongen/commongen.${TYPE}.cs_str.txt

