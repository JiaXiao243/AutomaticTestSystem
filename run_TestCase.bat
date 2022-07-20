set sed="C:\Program Files\Git\usr\bin\sed.exe"
set CUDA_VISIBLE_DEVICES=0
python -m pip install -r requirements.txt
rem python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
python -m pytest -sv test_ocr_acc.py  --alluredir=./result
copy environment/environment.properties_linux ./result
ren ./result/environment/environment.properties_linux ./result/environment/environment.properties
allure generate ./result/ -o ./report_test/ --clean
