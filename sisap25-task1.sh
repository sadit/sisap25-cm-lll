# docker build -t sisap25/diskann-image .
echo "RUN AS: bash -x task1.sh 2>&1 | tee log-task1.txt"

#PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025/without-gold
PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025/
#PATH_TO_CONTAINER_DIR=/app/DiskANN/data/
PATH_TO_CONTAINER_DIR=/data/
OUT_PATH_TO_HOST_DIR=$(pwd)/results-task1
OUT_PATH_TO_CONTAINER_DIR=/results

mkdir $OUT_PATH_TO_HOST_DIR
echo "==== pwd: $(pwd)"
echo "==== directory listing: "
ls
echo "==== environment"
set
echo "==== RUN BEGINS $(date)"
docker run \
    -it \
    --cpus=8 \
    --memory=16g \
    --volume $PATH_TO_HOST_DIR:$PATH_TO_CONTAINER_DIR:ro \
    --volume $OUT_PATH_TO_HOST_DIR:$OUT_PATH_TO_CONTAINER_DIR:rw \
      sisap25/diskann-image \
      python3 RunAndEval.py --dataname pubmed23 --results /results/results.csv


echo "==== RUN ENDS $(date)"


