
python -m pip install -r requirements.txt
which allure
rm -rf /usr/bin/allure
ln -s /workspace/AutomaticTestSystem/allure/bin/allure /usr/bin/allure

python -m pytest -sv test_ocr_acc_mac.py  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
allure generate ./result/ -o ./report_test/ --clean
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
