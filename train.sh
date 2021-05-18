python main.py --dataset shanghaitechpa \
--model CSRNet \
--train-files /data/CrowdCount/train.txt \
--val-files /data/CrowdCount/val.txt \
--gpu-devices 4 \
--lr 1e-5 \
--optim adam \
--loss mseloss \
--checkpoints ./checkpoints/demo \
--summary-writer ./runs/demo \
--train-batch 1