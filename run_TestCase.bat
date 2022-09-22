set sed="C:\Program Files\Git\usr\bin\sed.exe"
set CUDA_VISIBLE_DEVICES=0
python -m pip install -r requirements.txt
rem python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
python -m pytest -sv %1  --alluredir=./result
rem xcopy environment\environment.properties_win ./result /s /e /y /d
rem cd result
rem ren environment.properties_win environment.properties
rem cd ..
allure generate ./result/ -o ./report_test/ --clean
echo start_report_uploaded
set REPORT_SERVER=https://xly.bce.baidu.com/ipipe/ipipe-report
set REPORT_SERVER_USERNAME=%2
set REPORT_SERVER_PASSWORD=%3
curl -s %REPORT_SERVER%/report/upload.sh | bash -s report_test %4 result
echo report_uploaded
