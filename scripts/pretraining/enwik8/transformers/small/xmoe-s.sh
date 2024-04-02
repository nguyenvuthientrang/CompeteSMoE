mkdir -p checkpoints/enwik8/transformers-s/xmoe/pertubed_2/epsw_1e-3_epsx_0

args="
--data data/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsg \
--gate_name xmoe \
--nlayers 6 \
--hid-sz 264 \
--inner-hid-sz 264 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/enwik8/transformers-s/xmoe/pertubed_2/epsw_1e-3_epsx_0/xmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8