yes | conda create -n coco_score python=2.7
source ~/anaconda3/etc/profile.d/conda.sh
yes | conda activate coco_score
yes | pip install numpy
yes | pip install -U spacy
yes | python -m spacy download en_core_web_sm
yes | bash ~/CommonsenseReasoning/evaluation/Traditional/get_stanford_models.sh
