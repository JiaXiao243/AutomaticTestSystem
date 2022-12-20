# export HOMEBREW_BOTTLE_DOMAIN=''
# brew install allure
python -m pip install -r requirements.txt
which allure
rm -rf /usr/bin/allure
ln -s /workspace/AutomaticTestSystem/allure/bin/allure /usr/bin/allure

python -m pytest -sv $1  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
cp environment/environment.properties_mac ./result
mv ./result/environment.properties_mac ./result/environment.properties
allure generate ./result/ -o ./report_test/ --clean
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
set +x;
export REPORT_SERVER="https://xly.bce.baidu.com/ipipe/ipipe-report"
export REPORT_SERVER_USERNAME=$2
export REPORT_SERVER_PASSWORD=$3
curl -s ${REPORT_SERVER}/report/upload.sh | bash -s report_test $4 result
echo "report uploaded"
