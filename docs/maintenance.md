# Maintenance

```sh
# Compress experiment and data directories
tar -zcvf experiments-2021-12-24.tar.gz experiments
tar -zcvf data-2021-12-24.tar.gz data

# Upload to Nectar Containers
swift upload fourierflow experiments-2021-12-24.tar.gz --info -S 1073741824
swift upload fourierflow data-2021-12-24.tar.gz --info -S 1073741824

# Occasionally, we need to manually wandb cache size. Wandb doesn't clean up
# automatically
wandb artifact cache cleanup 1GB
```
