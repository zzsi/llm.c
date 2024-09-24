mpirun --allow-run-as-root -np 4 ./train_gpt2fp32cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o log124M \
    -b 8 \
    -t 1024 \
    -l 0.0006 \
    -v 250 \
    -s 20000 \
    -m 100


# Usage:   ./train_gpt2fp32cu [options]
# Options:
#   -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)
#   -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)
#   -o <string> output log file (default = NULL)
#   -b <int>    batch size B (default = 4)
#   -t <int>    sequence length T (default = 1024)
#   -l <float>  learning rate (default = 3e-4f)
#   -v <int>    val_loss_every, how often we evaluate val loss (default = 20)
#   -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)
#   -s <int>    sample_every, how often we inference the model (default = 20)
#   -g <int>    genT, how many steps of inference we do (default = 64)