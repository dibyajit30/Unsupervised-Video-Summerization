for seed in {1..10}
do
    python trainer.py --seed $seed  --run_name resnet50_seed_$seed --cnn_feat resnet50
done