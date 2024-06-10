# ray tune scripts

# TIME - hold
#nohup python -u main.py --mode tune --general_config time/hold/gine --ray_config GINE --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/digine --ray_config DIGINE --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_digine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigine --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_bigine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gcn --ray_config GCN --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/digcn --ray_config DIGCN --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigcn --ray_config BIGCN --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gat --ray_config GAT --device '3,4' --num_gpu_per_trial 0.3 >./ray_time_hold_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/digat --ray_config DIGAT --device '2,3' --num_gpu_per_trial 0.3 >./ray_time_hold_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigat --ray_config BIGAT --device '1,5' --num_gpu_per_trial 0.3 >./ray_time_hold_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gine --pe_config lap10/lap_spe --ray_config GINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_time_hold_gine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gine --pe_config lap10/lap_naive --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_time_hold_gine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_time_hold_gine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/gine --pe_config maglap10/maglap_1q_naive --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_time_hold_gine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_time_hold_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigine --pe_config lap10/lap_naive --ray_config BIGINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_time_hold_bigine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_time_hold_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/bigine --pe_config maglap10/maglap_1q_naive --ray_config BIGINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_time_hold_bigine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/msgnn --ray_config MSGNN --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_hold_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/diigcn --ray_config DIIGCN --device '0,1' --num_gpu_per_trial 0.3 >./ray_time_hold_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/dgcn --ray_config DGCN --device '0,1' --num_gpu_per_trial 0.3 >./ray_time_hold_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/performer_lap --pe_config lap10/lap_spe --ray_config PERFORMER --device '0,1' --num_gpu_per_trial 0.24 >./ray_time_hold_performer_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/performer_lap_naive --pe_config lap10/lap_naive --ray_config PERFORMER --device '3,4' --num_gpu_per_trial 0.24 >./ray_time_hold_performer_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/performer_maglap --pe_config maglap10/maglap_1q_spe --ray_config PERFORMER --device '6,7' --num_gpu_per_trial 0.24 >./ray_time_hold_performer_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/performer_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_time_hold_performer_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_time_hold_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_time_hold_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '5,6' --num_gpu_per_trial 0.24 >./ray_time_hold_diperformer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/hold/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_time_hold_biperformer_mag.log 2>&1 </dev/null &

# TIME - setup
#nohup python -u main.py --mode tune --general_config time/setup/gine --ray_config GINE --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_setup_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/digine --ray_config DIGINE --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_setup_digine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigine --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_setup_bigine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gcn --ray_config GCN --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_setup_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/digcn --ray_config DIGCN --device '6,7' --num_gpu_per_trial 0.3 >./ray_time_setup_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigcn --ray_config BIGCN --device '0,1' --num_gpu_per_trial 0.3 >./ray_time_setup_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gat --ray_config GAT --device '2' --num_gpu_per_trial 0.3 >./ray_time_setup_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/digat --ray_config DIGAT --device '3' --num_gpu_per_trial 0.3 >./ray_time_setup_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigat --ray_config BIGAT --device '5' --num_gpu_per_trial 0.3 >./ray_time_setup_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gine --pe_config lap10/lap_spe --ray_config GINE --device '2,3' --num_gpu_per_trial 0.3 >./ray_time_setup_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gine --pe_config lap10/lap_naive --ray_config GINE --device '4,5' --num_gpu_per_trial 0.3 >./ray_time_setup_gine_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '1,2' --num_gpu_per_trial 0.3 >./ray_time_setup_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/gine --pe_config maglap10/maglap_1q_naive --ray_config GINE --device '1' --num_gpu_per_trial 0.3 >./ray_time_setup_gine_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '0,1' --num_gpu_per_trial 0.3 >./ray_time_setup_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigine --pe_config lap10/lap_naive --ray_config BIGINE --device '2' --num_gpu_per_trial 0.3 >./ray_time_setup_bigine_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '2,3' --num_gpu_per_trial 0.3 >./ray_time_setup_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/bigine --pe_config maglap10/maglap_1q_naive --ray_config BIGINE --device '3' --num_gpu_per_trial 0.3 >./ray_time_setup_bigine_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.3 >./ray_time_setup_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/diigcn --ray_config DIIGCN --device '5,6' --num_gpu_per_trial 0.3 >./ray_time_setup_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/dgcn --ray_config DGCN --device '2,3' --num_gpu_per_trial 0.3 >./ray_time_setup_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/performer_lap --pe_config lap10/lap_spe --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.24 >./ray_time_setup_performer_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/performer_lap_naive --pe_config lap10/lap_naive --ray_config PERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_time_setup_performer_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/performer_maglap --pe_config maglap10/maglap_1q_spe --ray_config PERFORMER --device '4,5' --num_gpu_per_trial 0.24 >./ray_time_setup_performer_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/performer_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6,7' --num_gpu_per_trial 0.24 >./ray_time_setup_performer_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_time_setup_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '3,4' --num_gpu_per_trial 0.24 >./ray_time_setup_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_time_setup_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_time_setup_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_time_setup_diperformer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config time/setup/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_time_setup_biperformer_mag.log 2>&1 </dev/null &

# AMP - gain
#nohup python -u main.py --mode tune --general_config amp/gain/gin --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_gain_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/digin --ray_config DIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_gain_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigin --ray_config BIGIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gcn --ray_config GCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_gain_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/digcn --ray_config DIGCN --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_gain_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigcn --ray_config BIGCN --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gat --ray_config GAT --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_gain_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/digat --ray_config DIGAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_gain_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigat --ray_config BIGAT --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gin --pe_config lap10/lap_naive --ray_config GIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_gin_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gin --pe_config lap10/lap_spe --ray_config GIN --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_gain_gin_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gin --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_gain_gin_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gin --pe_config maglap10/maglap_1q_spe --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_gain_gin_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '1,3,4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigin --pe_config lap10/lap_naive --ray_config BIGIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigin_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 500 --general_config amp/gain/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '1,2,3,4,5,6' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigin --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigin_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_gain_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_gain_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/digps --pe_config lap10/lap_naive --ray_config DIGPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_digps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigps --pe_config lap10/lap_naive --ray_config BIGPS --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/digps_mag --pe_config maglap10/maglap_1q_naive --ray_config DIGPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_gain_digps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/bigps_mag --pe_config maglap10/maglap_1q_naive --ray_config BIGPS --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_gain_bigps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_amp_gain_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/diigcn --ray_config DIIGCN --device '4,5' --num_gpu_per_trial 0.33 >./ray_amp_gain_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/gain/dgcn --ray_config DGCN --device '4,5' --num_gpu_per_trial 0.33 >./ray_amp_gain_dgcn.log 2>&1 </dev/null &

# AMP - pm
#nohup python -u main.py --mode tune --general_config amp/pm/gin --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_pm_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/digin --ray_config DIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_pm_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigin --ray_config BIGIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gcn --ray_config GCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_pm_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/digcn --ray_config DIGCN --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_pm_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigcn --ray_config BIGCN --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gat --ray_config GAT --device '3,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/digat --ray_config DIGAT --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_pm_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigat --ray_config BIGAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_pm_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_pm_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_pm_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/digps --pe_config lap10/lap_naive --ray_config DIGPS --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_pm_digps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigps --pe_config lap10/lap_naive --ray_config BIGPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/digps_mag --pe_config maglap10/maglap_1q_naive --ray_config DIGPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_pm_digps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/bigps_mag --pe_config maglap10/maglap_1q_naive --ray_config BIGPS --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_pm_bigps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_amp_pm_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/diigcn --ray_config DIIGCN --device '6,7' --num_gpu_per_trial 0.33 >./ray_amp_pm_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/pm/dgcn --ray_config DGCN --device '6,7' --num_gpu_per_trial 0.33 >./ray_amp_pm_dgcn.log 2>&1 </dev/null &

# AMP - bw
#nohup python -u main.py --mode tune --general_config amp/bw/gin --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_bw_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/digin --ray_config DIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_bw_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/bigin --ray_config BIGIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gcn --ray_config GCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_bw_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/digcn --ray_config DIGCN --device '1,2' --num_gpu_per_trial 0.24 >./ray_amp_bw_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/bigcn --ray_config BIGCN --device '3,4' --num_gpu_per_trial 0.24 >./ray_amp_bw_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gat --ray_config GAT --device '1,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/digat --ray_config DIGAT --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_bw_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/bigat --ray_config BIGAT --device '1,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_bw_gine_lap.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_bw_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_bw_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '2,3' --num_gpu_per_trial 0.24 >./ray_amp_bw_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/digps --pe_config lap10/lap_naive --ray_config DIGPS --device '6,7' --num_gpu_per_trial 0.24 >./ray_amp_bw_digps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/bigps --pe_config lap10/lap_naive --ray_config BIGPS --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_bw_bigps.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/digps_mag --pe_config maglap10/maglap_1q_naive --ray_config DIGPS --device '4,5' --num_gpu_per_trial 0.24 >./ray_amp_bw_digps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/bigps_mag --pe_config maglap10/maglap_1q_naive --ray_config BIGPS --device '0,1' --num_gpu_per_trial 0.24 >./ray_amp_bw_bigps_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_amp_bw_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/diigcn --ray_config DIIGCN --device '0,1' --num_gpu_per_trial 0.33 >./ray_amp_bw_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config amp/bw/dgcn --ray_config DGCN --device '0,1' --num_gpu_per_trial 0.33 >./ray_amp_bw_dgcn.log 2>&1 </dev/null &


# HLS - cp
#nohup python -u main.py --mode tune --general_config hls/cp/gine --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/digine --ray_config DIGINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_hls_cp_digine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --ray_config BIGINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gcn --ray_config GCN --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/digcn --ray_config DIGCN --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_cp_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigcn --ray_config BIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gat --ray_config GAT --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/digat --ray_config DIGAT --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigat --ray_config BIGAT --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config lap10/lap_spe --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '3,4' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config lap10/lap_spe --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config lap10/lap_naive --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gine --pe_config maglap10/maglap_1q_naive --ray_config GINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_gine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config lap10/lap_naive --ray_config BIGINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/bigine --pe_config maglap10/maglap_1q_naive --ray_config BIGINE --device '0' --num_gpu_per_trial 0.24 >./ray_hls_cp_bigine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_hls_cp_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/diigcn --ray_config DIIGCN --device '2,3' --num_gpu_per_trial 0.33 >./ray_hls_cp_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/dgcn --ray_config DGCN --device '2,3' --num_gpu_per_trial 0.33 >./ray_hls_cp_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '0' --num_gpu_per_trial 0.49 >./ray_hls_cp_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '0' --num_gpu_per_trial 0.49 >./ray_hls_cp_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '0' --num_gpu_per_trial 0.49 >./ray_hls_cp_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/cp/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '0' --num_gpu_per_trial 0.49 >./ray_hls_cp_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &

# HLS - dsp
#nohup python -u main.py --mode tune --general_config hls/dsp/gine --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/digine --ray_config DIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_dsp_digine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigine --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gcn --ray_config GCN --device '2,3' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/digcn --ray_config DIGCN --device '3,4' --num_gpu_per_trial 0.24 >./ray_hls_dsp_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigcn --ray_config BIGCN --device '4,5' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gat --ray_config GAT --device '4,5' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/digat --ray_config DIGAT --device '1' --num_gpu_per_trial 0.24 >./ray_hls_dsp_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigat --ray_config BIGAT --device '2' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gine --pe_config lap10/lap_spe --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gine --pe_config lap10/lap_naive --ray_config GINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gine --pe_config maglap10/maglap_1q_naive --ray_config GINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_hls_dsp_gine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigine --pe_config lap10/lap_naive --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/bigine --pe_config maglap10/maglap_1q_naive --ray_config BIGINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_hls_dsp_bigine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_hls_dsp_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/diigcn --ray_config DIIGCN --device '0,1' --num_gpu_per_trial 0.33 >./ray_hls_dsp_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/dgcn --ray_config DGCN --device '6,7' --num_gpu_per_trial 0.33 >./ray_hls_dsp_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '2,3,4' --num_gpu_per_trial 0.49 >./ray_hls_dsp_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '1,2' --num_gpu_per_trial 0.49 >./ray_hls_dsp_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '5,6,7' --num_gpu_per_trial 0.49 >./ray_hls_dsp_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/dsp/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '4,5' --num_gpu_per_trial 0.49 >./ray_hls_dsp_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &

# HLS - lut
#nohup python -u main.py --mode tune --general_config hls/lut/gine --ray_config GINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_hls_lut_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/digine --ray_config DIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_lut_digine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigine --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gcn --ray_config GCN --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_lut_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/digcn --ray_config DIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_lut_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigcn --ray_config BIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gat --ray_config GAT --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_lut_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/digat --ray_config DIGAT --device '4' --num_gpu_per_trial 0.24 >./ray_hls_lut_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigat --ray_config BIGAT --device '5' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.33 >./ray_hls_lut_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/diigcn --ray_config DIIGCN --device '2,3' --num_gpu_per_trial 0.33 >./ray_hls_lut_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/dgcn --ray_config DGCN --device '4,5' --num_gpu_per_trial 0.33 >./ray_hls_lut_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gine --pe_config lap10/lap_spe --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_hls_lut_gine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gine --pe_config lap10/lap_naive --ray_config GINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_hls_lut_gine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gine --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_hls_lut_gine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gine --pe_config maglap10/maglap_1q_naive --ray_config GINE --device '2,4' --num_gpu_per_trial 0.24 >./ray_hls_lut_gine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigine --pe_config lap10/lap_spe --ray_config BIGINE --device '1,2' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigine_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigine --pe_config lap10/lap_naive --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigine_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigine --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigine_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/bigine --pe_config maglap10/maglap_1q_naive --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_hls_lut_bigine_maglap10_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gps_lap --pe_config lap10/lap_spe --ray_config GPS --device '3,4' --num_gpu_per_trial 0.49 >./ray_hls_lut_gps_lap10_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gps_lap_naive --pe_config lap10/lap_naive --ray_config GPS --device '2,3' --num_gpu_per_trial 0.49 >./ray_hls_lut_gps_lap10_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gps_maglap --pe_config maglap10/maglap_1q_spe --ray_config GPS --device '6,7' --num_gpu_per_trial 0.49 >./ray_hls_lut_gps_maglap10_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config hls/lut/gps_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GPS --device '6,7' --num_gpu_per_trial 0.49 >./ray_hls_lut_gps_maglap10_maglap_1q_naive.log 2>&1 </dev/null &

# CG - cpu
#nohup python -u main.py --mode tune --general_config cg/cpu/gin --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/digin --ray_config DIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_cpu_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigin --ray_config BIGIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gcn --ray_config GCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/digcn --ray_config DIGCN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigcn --ray_config BIGCN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gat --ray_config GAT --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/digat --ray_config DIGAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_cg_cpu_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigat --ray_config BIGAT --device '2,5' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gin --pe_config lap10/lap_naive --ray_config GIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/gin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_cpu_gin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigin_lap_naive --pe_config lap10/lap_naive --ray_config BIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/bigin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bigin_maglap_1q_naive.log 2>&1 </dev/null &
##nohup python -u main.py --mode tune --general_config cg/cpu/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.24 >./ray_cg_cpu_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/diigcn --ray_config DIIGCN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_cpu_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/cpu/dgcn --ray_config DGCN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_cpu_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/performer --pe_config lap10/lap_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.24 >./ray_cg_cpu_performer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_cg_cpu_dierformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '4' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bierformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/performer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.24 >./ray_cg_cpu_performer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_cg_cpu_dierformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/cpu/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '4' --num_gpu_per_trial 0.24 >./ray_cg_cpu_bierformer.log 2>&1 </dev/null &

# CG - gpu640
#nohup python -u main.py --mode tune --general_config cg/gpu640/gin --ray_config GIN --device '3,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/digin --ray_config DIGIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigin --ray_config BIGIN --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gcn --ray_config GCN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/digcn --ray_config DIGCN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigcn --ray_config BIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gat --ray_config GAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/digat --ray_config DIGAT --device '1,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigat --ray_config BIGAT --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '0,3' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gin --pe_config lap10/lap_naive --ray_config GIN --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/gin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_gin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigin_lap_naive --pe_config lap10/lap_naive --ray_config BIGIN --device '2,3,4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/bigin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bigin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/diigcn --ray_config DIIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu640/dgcn --ray_config DGCN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/performer --pe_config lap10/lap_naive --ray_config PERFORMER --device '1' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_performer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '2' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '3' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/performer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_performer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_dierformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu640/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '4' --num_gpu_per_trial 0.24 >./ray_cg_gpu640_bierformer.log 2>&1 </dev/null &

# CG - gpu630
#nohup python -u main.py --mode tune --general_config cg/gpu630/gin --ray_config GIN --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gine.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/digin --ray_config DIGIN --device '3,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigin --ray_config BIGIN --device '2,3' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gcn --ray_config GCN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/digcn --ray_config DIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigcn --ray_config BIGCN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gat --ray_config GAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/digat --ray_config DIGAT --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigat --ray_config BIGAT --device '3,4' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gin --pe_config lap10/lap_naive --ray_config GIN --device '0' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '1' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/gin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '1,2' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_gin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '3,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigin_lap_naive --pe_config lap10/lap_naive --ray_config BIGIN --device '6,7' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '3,4' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/bigin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '5,6' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bigin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/diigcn --ray_config DIIGCN --device '4,5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config cg/gpu630/dgcn --ray_config DGCN --device '0,1' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/performer --pe_config lap10/lap_naive --ray_config PERFORMER --device '5' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_performer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '2' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '3' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/performer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_performer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '7' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_dierformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --num_trial 50 --general_config cg/gpu630/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '4' --num_gpu_per_trial 0.24 >./ray_cg_gpu630_bierformer.log 2>&1 </dev/null &

# SR - shared
#nohup python -u main.py --mode tune --general_config sr/shared/gin --ray_config GIN --device '0' --num_gpu_per_trial 0.20 >./ray_sr_shared_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/digin --ray_config DIGIN --device '1' --num_gpu_per_trial 0.20 >./ray_sr_shared_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigin --ray_config BIGIN --device '2' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gcn --ray_config GCN --device '1' --num_gpu_per_trial 0.20 >./ray_sr_shared_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/digcn --ray_config DIGCN --device '4' --num_gpu_per_trial 0.20 >./ray_sr_shared_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigcn --ray_config BIGCN --device '5' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gat --ray_config GAT --device '2' --num_gpu_per_trial 0.20 >./ray_sr_shared_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/digat --ray_config DIGAT --device '3,4' --num_gpu_per_trial 0.20 >./ray_sr_shared_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigat --ray_config BIGAT --device '1,2' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '3' --num_gpu_per_trial 0.20 >./ray_sr_shared_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gin --pe_config lap10/lap_naive --ray_config GIN --device '4' --num_gpu_per_trial 0.20 >./ray_sr_shared_gin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '5' --num_gpu_per_trial 0.20 >./ray_sr_shared_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/gin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '0' --num_gpu_per_trial 0.20 >./ray_sr_shared_gin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '2' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigin_lap_naive --pe_config lap10/lap_naive --ray_config BIGIN --device '3' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '4' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/bigin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '5' --num_gpu_per_trial 0.20 >./ray_sr_shared_bigin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.20 >./ray_sr_shared_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/diigcn --ray_config DIIGCN --device '6,7' --num_gpu_per_trial 0.20 >./ray_sr_shared_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/dgcn --ray_config DGCN --device '4,5' --num_gpu_per_trial 0.20 >./ray_sr_shared_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/performer --pe_config lap10/lap_naive --ray_config PERFORMER --device '4' --num_gpu_per_trial 0.20 >./ray_sr_shared_performer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.20 >./ray_sr_shared_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.20 >./ray_sr_shared_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/performer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '4' --num_gpu_per_trial 0.20 >./ray_sr_shared_performer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.20 >./ray_sr_shared_diperformer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/shared/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.20 >./ray_sr_shared_biperformer_mag.log 2>&1 </dev/null &


# SR - root
#nohup python -u main.py --mode tune --general_config sr/root/gin --ray_config GIN --device '1' --num_gpu_per_trial 0.20 >./ray_sr_root_gin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/digin --ray_config DIGIN --device '5' --num_gpu_per_trial 0.20 >./ray_sr_root_digin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigin --ray_config BIGIN --device '6' --num_gpu_per_trial 0.20 >./ray_sr_root_bigin.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gcn --ray_config GCN --device '3' --num_gpu_per_trial 0.20 >./ray_sr_root_gcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/digcn --ray_config DIGCN --device '0' --num_gpu_per_trial 0.20 >./ray_sr_root_digcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigcn --ray_config BIGCN --device '4' --num_gpu_per_trial 0.20 >./ray_sr_root_bigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gat --ray_config GAT --device '5' --num_gpu_per_trial 0.20 >./ray_sr_root_gat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/digat --ray_config DIGAT --device '1,2' --num_gpu_per_trial 0.20 >./ray_sr_root_digat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigat --ray_config BIGAT --device '3,4' --num_gpu_per_trial 0.20 >./ray_sr_root_bigat.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gine_lap --pe_config lap10/lap_spe --ray_config GINE --device '0' --num_gpu_per_trial 0.20 >./ray_sr_root_gine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gin --pe_config lap10/lap_naive --ray_config GIN --device '1' --num_gpu_per_trial 0.20 >./ray_sr_root_gin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gine_maglap --pe_config maglap10/maglap_1q_spe --ray_config GINE --device '3' --num_gpu_per_trial 0.20 >./ray_sr_root_gine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/gin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config GIN --device '5' --num_gpu_per_trial 0.20 >./ray_sr_root_gin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigine_lap --pe_config lap10/lap_spe --ray_config BIGINE --device '0' --num_gpu_per_trial 0.20 >./ray_sr_root_bigine_lap_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigin_lap_naive --pe_config lap10/lap_naive --ray_config BIGIN --device '4' --num_gpu_per_trial 0.20 >./ray_sr_root_bigin_lap_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigine_maglap --pe_config maglap10/maglap_1q_spe --ray_config BIGINE --device '6' --num_gpu_per_trial 0.20 >./ray_sr_root_bigine_maglap_1q_spe.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/bigin_maglap_naive --pe_config maglap10/maglap_1q_naive --ray_config BIGIN --device '5' --num_gpu_per_trial 0.20 >./ray_sr_root_bigin_maglap_1q_naive.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/msgnn --ray_config MSGNN --device '0' --num_gpu_per_trial 0.20 >./ray_sr_root_msgnn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/diigcn --ray_config DIIGCN --device '0,1' --num_gpu_per_trial 0.20 >./ray_sr_root_diigcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/dgcn --ray_config DGCN --device '0,1' --num_gpu_per_trial 0.20 >./ray_sr_root_dgcn.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/performer --pe_config lap10/lap_naive --ray_config PERFORMER --device '4' --num_gpu_per_trial 0.20 >./ray_sr_root_performer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/diperformer --pe_config lap10/lap_naive --ray_config PERFORMER --device '5' --num_gpu_per_trial 0.20 >./ray_sr_root_diperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/biperformer --pe_config lap10/lap_naive --ray_config BIPERFORMER --device '6' --num_gpu_per_trial 0.20 >./ray_sr_root_biperformer.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/performer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '4' --num_gpu_per_trial 0.20 >./ray_sr_root_performer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/diperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config PERFORMER --device '6' --num_gpu_per_trial 0.20 >./ray_sr_root_diperformer_mag.log 2>&1 </dev/null &
#nohup python -u main.py --mode tune --general_config sr/root/biperformer_mag --pe_config maglap10/maglap_1q_naive --ray_config BIPERFORMER --device '7' --num_gpu_per_trial 0.20 >./ray_sr_root_biperformer_mag.log 2>&1 </dev/null &