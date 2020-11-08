yes|conda create -n pivot_score python=3.6
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pivot_score
yes|pip install spacy
yes|python -m spacy download en
yes|pip install networkx
