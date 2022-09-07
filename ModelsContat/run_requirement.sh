# repo_list='PaddleClas PaddleOCR PaddleDetection PaddleSeg PaddleSpeech PaddleNLP PGL PaddleScience FastDeploy'
set -x
repo_list='PaddleClas PaddleOCR PaddleDetection PaddleSeg PaddleNLP PaddleSpeech'
for repo in $repo_list
do
echo $repo
git clone http://github.com/PaddlePaddle/$repo.git
cd  $repo
if [ "$repo"! = "PaddleSpeech" ];then
python -m pip install -r requirements.txt
fi
python -m pip install .
cd ..
done

