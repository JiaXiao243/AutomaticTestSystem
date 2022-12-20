# paddle
echo "paddle_version:"
python -c 'import paddle; print(paddle.__version__, paddle.version.commit)'
# ubuntu
if [ -f "/etc/lsb-release" ];then
cat /etc/lsb-release
apt-get update
apt-get install curl -y
apt-get install -y nodejs
ln -s /usr/bin/nodejs /usr/bin/node
apt install -y openjdk-8-jdk
export PATH=/usr/bin/allure:$PATH

elif [ -f "/etc/redhat-release" ];then
cat /etc/redhat-release
# centos
yum update -y > /dev/null
yum install curl -y
yum install -y nodejs > /dev/null
yum install -y java-1.8.0-openjdk-devel.x86_64 > /dev/null
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
export JRE_HOME=$JAVA_HOME/jre
export CLASSPATH=$JAVA_HOME/lib:$JRE_HOME/lib:$CLASSPATH
export PATH=$JAVA_HOME/bin:$JRE_HOME/bin:$PATH
export PATH=/usr/bin/allure:$PATH

else
# mac
echo "mac_system"
export HOMEBREW_BOTTLE_DOMAIN=''
brew install allure
fi

rm -rf /usr/bin/allure
ln -s /ssd2/jiaxiao01/AutomaticTestSystem/allure/bin/allure /usr/bin/allure
python -m pip install --ignore-installed --user -r requirements.txt

# export CUDA_VISIBLE_DEVICES=0,1
which allure

 python -m pytest -sv $1  --alluredir=./result #--alluredir用于指定存储测试结果的路径)
exit_code=$?
echo 'exit_code:'$exit_code
cp environment/environment.properties_linux ./result 
mv ./result/environment.properties_linux ./result/environment.properties
allure generate ./result/ -o ./report_test/ --clean
set +x;
export REPORT_SERVER="https://xly.bce.baidu.com/ipipe/ipipe-report"
export REPORT_SERVER_USERNAME=$2
export REPORT_SERVER_PASSWORD=$3
curl -s ${REPORT_SERVER}/report/upload.sh | bash -s report_test $4 result
echo "report uploaded"

exit $exit_code
# python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
