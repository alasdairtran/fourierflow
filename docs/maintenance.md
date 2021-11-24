# Maintenance

```sh
# Compress experiment and data directories
tar -zcvf experiments.tar.gz experiments
tar -zcvf data.tar.gz data

# Extract the archives
tar -zxvf experiments.tar.gz
tar -zxvf data.tar.gz

# Upload to Nectar Containers
swift upload fourierflow experiments.tar.gz --info
swift upload fourierflow data.tar.gz --info

# Occasionally, we need to manually wandb cache size. Wandb doesn't clean up
# automatically
wandb artifact cache cleanup 1GB
```
