### Incremental UPL
python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400

python3 main.py --model_name stam --dataset svhn --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000

python3 main.py --model_name stam --dataset cifar-10 --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb --delta 2500 
  
python3 main.py --model_name stam --dataset emnist --N_l 10 --N_e 100 --N_p 2000 --ntrials 3 --ntp 5 --log stam_incremental --load_log stam_incremental --schedule_flag 1 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400


### Uniform UPL
#python3 main.py --model_name stam --dataset mnist --N_l 10 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400 

#python3 main.py --model_name stam --dataset svhn --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 2000

#python3 main.py --model_name stam --dataset cifar-10 --N_l 100 --N_e 100 --N_p 10000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format rgb  --delta 2500

#python3 main.py --model_name stam --dataset emnist --N_l 10 --N_e 100 --N_p 2000 --ntrials 3 --ntp 5 --log stam_uniform --load_log stam_uniform --schedule_flag 2 --shuffle_flag 1 --scale_flag --layers_flag 1 --norm_flag 6 --color_format gray --delta 400
