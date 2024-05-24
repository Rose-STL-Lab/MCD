mkdir temp
cd temp

wget https://github.com/sakhanna/SRU_for_GCI/archive/refs/heads/master.zip
unzip master.zip

if [ ! -d "../data/dream3/raw" ]
then
    mkdir -p ../data/dream3/raw
fi

mv SRU_for_GCI-master/data/dream3/* ../data/dream3/raw/
cd ../
rm -r temp
python3 src/utils/data_gen/process_dream3.py --dataset_dir data/dream3/raw --save_dir data/dream3/

