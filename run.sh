#!/bin/sh

DATETIME=`date +%s`
FILENAME=`date -r ${DATETIME} +"%Y-%m-%d_%H-%M-%S"`

# フォルダ作成
echo "mkdir ./parameter"
mkdir ./parameter
echo "mkdir ./${FILENAME}"
mkdir "./${FILENAME}"

# パラメータ生成
echo "python3 make_batch_file.py"
python3 make_batch_file.py

# モデル実行
echo "python3 run.py"
python3 run.py ./${FILENAME}/

# 結果のプロット
echo "python3 my_model_plot.py"
python3 plot.py ./${FILENAME}/

# 結果の保存
echo "mv ./parameter ./${FILENAME}/parameter"
mv ./parameter ./${FILENAME}/parameter

# プログラムの保存
echo "cp ./run.py ./${FILENAME}/run.py"
cp ./run.py ./${FILENAME}/run.py
echo "cp ./model_helper.py ./${FILENAME}/model_helper.py"
cp ./model_helper.py ./${FILENAME}/model_helper.py
echo "cp ./config.py ./${FILENAME}/config.py"
cp ./config.py ./${FILENAME}/config.py
echo "cp ./plot.py ./${FILENAME}/plot.py"
cp ./plot.py ./${FILENAME}/plot.py
echo "cp ./make_batch_file.py ./${FILENAME}/make_batch_file.py"
cp ./make_batch_file.py ./${FILENAME}/make_batch_file.py


echo "mv ./${FILENAME} ./result/${FILENAME}"
mv ./${FILENAME} ./result/${FILENAME}