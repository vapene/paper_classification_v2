nohup python try1.py --models "resnet" --epochs 150 --trial 1 > resnet.out &
sleep 1h
nohup python try1.py --models "alex" --epochs 150 --trial 1 > alexnet.out &
sleep 1h
nohup python try1.py --models "vgg" --epochs 150 --trial 1 > vgg11.out &
sleep 1h
nohup python try1.py --models "dense" --epochs 150 --trial 1 > densenet.out &
