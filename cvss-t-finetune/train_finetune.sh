#!/bin/bash
# Use Windows Python environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

LANG=es
DATA_ROOT=C:/Original\ Streamspeech\ -\ Copy/cvss-t-finetune
DATA=$DATA_ROOT/${LANG}-en/fbank2unit
model=streamspeech.finetuned.${LANG}-en

python fairseq/train.py $DATA \
  --user-dir researches/ctc_unity \
  --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
  --task speech_to_speech_ctc --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit_2pass_ctc_asr_st --label-smoothing 0.1 --rdrop-alpha 0.0 \
  --arch streamspeech --share-decoder-input-output-embed \
  --encoder-layers 12 --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4 \
  --translation-decoder-layers 4 --synthesizer-encoder-layers 2 \
  --decoder-layers 2  --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 8 \
  --k1 0 --k2 0 --n1 1 --n2 -1 \
  --chunk-size 999999 \
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1 \
  --train-subset train --valid-subset dev \
  --ctc-upsample-rate 25 \
  --save-dir checkpoints/$model \
  --validate-interval 200 --validate-interval-updates 200 \
  --save-interval 1 --save-interval-updates 100 \
  --keep-last-epochs 10 \
  --no-progress-bar --log-format json --log-interval 20 \
  --lr 0.0001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-8 --warmup-updates 2000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 1.0 \
  --max-tokens 12000 --max-target-positions 1200 --update-freq 2 \
  --attn-type espnet --pos-enc-type rel_pos \
  --keep-interval-updates 20 \
  --keep-best-checkpoints 10 \
  --seed 1 --fp16 --num-workers 4 \
  --reset-optimizer --reset-dataloader --reset-meters
