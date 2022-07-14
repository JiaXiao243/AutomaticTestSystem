# rm -rf PaddleOCR/rec_*
python -m pip install -r requirements.txt
# export CUDA_VISIBLE_DEVICES=0,1
python -m pytest -sv test_ocr_acc_report.py  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
allure generate ./result/ -o ./report_test/ --clean
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
