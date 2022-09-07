python -m pip install --upgrade parl opencv-python
repo_list='PaddleClas PaddleOCR PaddleDetection PaddleSeg PaddleSpeech PaddleNLP PGL PaddleScience FastDeploy'
for repo in $repo_list
do
echo $repo
git clone http://github.com/PaddlePaddle/$repo.git
cd  $repo
python -m pip install -r requirements.txt
cd ..
done


