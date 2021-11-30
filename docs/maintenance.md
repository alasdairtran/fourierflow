# Maintenance

```sh
# Compress experiment and data directories
tar -zcvf experiments.tar.gz experiments
tar -zcvf data.tar.gz data

# Upload to Nectar Containers
swift upload fourierflow experiments.tar.gz --info -S 1073741824
swift upload fourierflow data.tar.gz  --info -S 1073741824

# Occasionally, we need to manually wandb cache size. Wandb doesn't clean up
# automatically
wandb artifact cache cleanup 1GB
```
