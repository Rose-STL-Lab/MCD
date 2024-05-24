mkdir temp
cd temp

wget https://www.fmrib.ox.ac.uk/datasets/netsim/sims.tar.gz
tar xvf sims.tar.gz

if [ ! -d "../data/netsim/raw" ]
then
    mkdir -p ../data/netsim/raw
fi


mv *.mat ../data/netsim/raw/
cd ../
rm -r temp
python3 src/utils/data_gen/process_netsim.py --dataset_dir data/netsim/raw --save_dir data/netsim/

