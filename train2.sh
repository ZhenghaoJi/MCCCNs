python main.py --dataset shanghaitechpa \
--model MCCCN \
--train-files /data/CrowdCount/train.txt \
--val-files /data/CrowdCount/val.txt \
--gpu-devices 4 \
--lr 1e-4 \
--optim adam \
--loss mseloss \
--checkpoints ./checkpoints/demo \
--summary-writer ./runs/demo \
--train-batch 1 