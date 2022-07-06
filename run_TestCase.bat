set sed="C:\Program Files\Git\usr\bin\sed.exe"
export CUDA_VISIBLE_DEVICES=0
python -m pip install -r requirements.txt
python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
