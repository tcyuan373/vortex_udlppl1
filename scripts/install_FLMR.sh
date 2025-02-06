pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --user ujson gitpython easydict ninja datasets transformers pybind11

#FLMR
git clone https://github.com/LinWeizheDragon/FLMR.git
cd FLMR
pip install --user . 

#ColBERT
cd third_party/ColBERT
pip install  --user .
cd ../../..
rm -rf FLMR


python prepare_pipeline1.py