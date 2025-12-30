@echo off
py -3.12 scripts\convert_dicom_to_jpeg.py --input-dir "C:\Users\Akram\Downloads\ISIC_2020_Train_DICOM_corrected\train" --output-dir "data\train" --workers 8
pause
