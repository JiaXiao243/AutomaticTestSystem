rm -rf PaddleOCR/rec_*

python -m  pytest -sv test_ocr_acc.py --html=rec_report.html --capture=tee-sys
