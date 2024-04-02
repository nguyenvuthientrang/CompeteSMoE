mkdir -p checkpoints/enwik8/transformers-m/xmoe/pertubed_denom/eps1_1e-4_1e-3/

args="
--data data/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsg \
--gate_name xmoe \
--nlayers 8 \
--hid-sz 352 \
--inner-hid-sz 352 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 4000 \
--niter 80 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/enwik8/transformers-m/xmoe/pertubed_denom/eps1_1e-4_1e-3/xmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8