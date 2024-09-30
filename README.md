# Horticultural Temporal Fruit Monitoring via 3D Instance Segmentation and Re-Identification using Point Clouds

```
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Install MinkowskiEngine

If you don't have it, it's better to install ninja to compile MinkowskiEngine
```
pip install ninja
```

```
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install
```

## Instance Segmentation
```
export PYTHONPATH=$(realpath instance_segmentation)
cd instance_segmentation/mink_pan
python scripts/run_full_row.py --modelpath {PATH_TO_MODEL}
```

## Re-Identification
```
cd re_identification
python3 associate.py --data {PATH_TO_DATA_FOLDER} --iou 0.05 
python3 extractor.py --data {PATH_TO_DATA_FOLDER} --iou 0.05
python3 train.py --data {PATH_TO_DATA_FOLDER} --iou 0.05 --mode testinst
```


## Visualize Results
```
visualize entire test set in 2D
python3 visualize_clouds.py --data {PATH_TO_DATA_FOLDER} --iou 0.05

visualize entire test set in 2D
python3 visualize_clouds3D.py --data {PATH_TO_DATA_FOLDER} --iou 0.05

```