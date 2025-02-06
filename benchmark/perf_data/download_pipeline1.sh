mkdir pipeline1
mkdir pipeline1/models
mkdir pipeline1/data
mkdir pipeline1/index

wget https://vortexstorage7348269.blob.core.windows.net/flmrmodels/models_pipeline1.zip
unzip models_pipeline1.zip
mv models_pipeline1/* pipeline1/models
rm -rf models_pipeline1.zip models_pipeline1

wget wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_data.zip
unzip EVQA_data.zip
mv EVQA_data pipeline1/data
rm EVQA_data.zip

wget https://vortexstorage7348269.blob.core.windows.net/flmrdata/EVQA_passages.zip
unzip EVQA_passages.zip
mv EVQA_passages pipeline1/data
rm EVQA_passages.zip
