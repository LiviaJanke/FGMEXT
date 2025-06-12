# FGMEXT

#### chris_test_branch of LiviaJanke/FGMEXT

Fluxgate Magnetometer Extended Mode Data processing

How to Use:
1. date and spacecraft selection parameters plus file paths in `params.txt`
2. calibrated data and metadata file production
3. processing on alsvid
4. validation versus SPIN data

History of the processing is documented at the [wiki](https://github.com/LiviaJanke/FGMEXT/wiki)
 
The format of the `params.txt` file shall be:
```
# params.txt
# specifiy the parameters for the processing and file paths
craft = C1
date_entry =  20010324
path_lib = ./Lib/
path_cal = /Volumes/cluster/calibration/
path_bs  = /Volumes/cluster/bs/
path_out = /Volumes/cluster/extm/
```
