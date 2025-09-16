To run always use:

#export PYTHONPATH=$(pwd)
#python scripts/train_embed_only.py --csv data/combined_emails.csv --use_signals


export PYTHONPATH=$(pwd)
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export TOKENIZERS_PARALLELISM=true
export SBERT_BATCH=256
python scripts/train_embed_only.py --csv data/combined_emails.csv --use_signals --alpha 0.1
