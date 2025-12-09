# lightning-YOLOs

Download DOTA v1.5 dataset and display directory tree
```bash
wget https://www.ultralytics.com/assets/DOTAv1.5.zip
rm -rf DOTAv1.5 sample_data
unzip -qq DOTAv1.5.zip
python -m py_tree DOTAv1.5 -d 1
```