#!/bin/bash



for i in {1..10}
do
  
  echo "Run $i / 10"

  #python main.py --dataset wikipedia --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 14 --heads 8 --nums_layers 2 --dropout 0.2 --lr 5e-4 --epochs 35 --gpu 1
    python main.py --dataset wikipedia --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 14 --heads 8 --nums_layers 2 --dropout 0.2 --lr 5e-4 --epochs 35 --gpu 1

  #python main.py --dataset alpha --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 20 --heads 4 --dropout 0.2 --lr 5e-4 --epochs 35 

  #python main.py --dataset otc --num_neighbors 15 10  --hidden_channels 128 --num_timeslots 10 --heads 8 --dropout 0.2 --lr 5e-4 --epochs 35 --gpu 1

  #python main.py --dataset mooc --num_neighbors 15 10 --hidden_channels 64 --num_timeslots 3 --heads 8 --dropout 0.7 --lr 1e-4 --epochs 35 


done 