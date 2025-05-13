# FGMEXT

#### Forked from LiviaJanke/FGMEXT to cc87eof/FGMEXT on 25-Apr-25

Fluxgate Magnetometer Extended Mode Data processing

How to Use:

Download FGM_EXT_BS_to_CAL.py and Lib folder from this reposity https://github.com/LiviaJanke/FGMEXT.git  

 

Download BS_Raw_Files  and calibration folders from Alsvid:  
scp -r  lme19@alsvid.sp.ph.ic.ac.uk:/home/lme19/BS_Raw_Files .
 
scp -r  lme19@alsvid.sp.ph.ic.ac.uk:/home/lme19/calibration .
 

Create output folders named C1_EXT_Calibrated, C2_EXT_Calibrated, C3_EXT_Calibrated, C4_EXT_Calibrated 

 

Change lines 44 – 54 of FGM_EXT_BS_to_CAL.py to match the locations of your Lib, BS_Raw_Files, calibration, and output folders. 

 

 

Update 'Craft' and 'Index' variables at lines 28 and 30 respectively to pick a specific numbered EXT entry for a specific craft (generally index values range between 0 and 900) 

 

Run file – deposits calibrated data and metadata files into corresponding spacecraft output folder. 

 
