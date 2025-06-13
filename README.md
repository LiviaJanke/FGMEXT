# FGMEXT

#### chris_test_branch of LiviaJanke/FGMEXT

Fluxgate Magnetometer Extended Mode Data processing

How to Use:
1. for local setup include file paths in `paths.txt`
2. select spacecraft and date at the top of the .py file
3. check EXTM entry/exit times
4. check BS file decoded
5. calibrated data and metadata file production
6. processing on alsvid
7. validation versus SPIN data

History of the processing is documented at the [wiki](https://github.com/LiviaJanke/FGMEXT/wiki)
 
The format of the `paths.txt` file shall be:
```
# paths.txt
# specifiy the file paths
path_lib = ./Lib/
path_cal = /Volumes/cluster/calibration/
path_bs  = /Volumes/cluster/bs/
path_out = /Volumes/cluster/extm/
```
