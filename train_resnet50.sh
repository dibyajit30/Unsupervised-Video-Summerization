for seed in {1..5}
do
    python trainer.py --seed $seed  --run_name resnet50_gru_seed_$seed --cnn_feat resnet50
done
