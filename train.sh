./train_gpt2fp32cu \
    -i "dev/data/fineweb10B/fineweb_train_*.bin" \
    -j "dev/data/fineweb10B/fineweb_val_*.bin" \
    -o log124M \
    -t 1024 \
    -l 0.0006 \
    -v 250 \
    -s 20000 \
    -m 100


# sudo <CudaInstaller>.run --silent --driver
# -b 64 \