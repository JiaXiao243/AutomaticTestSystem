# rm -rf PaddleOCR/rec_*
apt-get update
apt install -y openjdk-8-jdk
python -m pip install -r requirements.txt
# export CUDA_VISIBLE_DEVICES=0,1
python -m pytest -sv test_ocr_acc.py  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
chmod -R 777 allure
pwd
which allure
ln -s /workspace/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
allure generate ./result/ -o ./report_test/ --clean
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
