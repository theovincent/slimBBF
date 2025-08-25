SHARED_ARGS="--features 64 128 128 2048 --replay_buffer_capacity 110_000 --batch_size 32 --min_update_horizon 3 --max_update_horizon 10 \
    --min_gamma 0.97 --max_gamma 0.997 --learning_rate 0.0001 \
    --horizon 27_000 --n_sampling_steps 100_000 --update_to_data 2 --n_initial_samples 2_000 \
    --tau 0.005 --epsilon_end 0.0 --epsilon_duration 2_001 --n_bins 51 \
    --gamma_horizon_decay_steps 10_000 --reset_frequency 20_000"

GAME="BattleZone"

PLATFORM="normal/local"  # nhrfau/cluster normal/cluster normal/local

SHARED_NAME="BBF"

BBF_ARGS="--experiment_name ${SHARED_NAME}_${GAME}"
launch_job/atari/${PLATFORM}_bbf.sh --first_seed 1 --last_seed 3 $SHARED_ARGS $BBF_ARGS
