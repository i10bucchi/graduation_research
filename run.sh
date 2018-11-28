#!/bin/sh

DATETIME=`date +%s`
FILENAME=`date -r ${DATETIME} +"%Y-%m-%d_%H-%M-%S"`

# フォルダ作成
echo "mkdir ./parameter"
mkdir ./parameter
echo "mkdir ./${FILENAME}"
mkdir "./${FILENAME}"

# パラメータ生成
echo "python make_batch_file.py"
python make_batch_file.py

# モデル実行
echo "python run_my_model.py"
python run_my_model.py ./${FILENAME}/

# 結果のプロット
echo "python my_model_plot.py"
python my_model_plot.py ./${FILENAME}/

# 結果の保存
echo "mv ./parameter ./${FILENAME}/parameter"
mv ./parameter ./${FILENAME}/parameter

# プログラムの保存
echo "cp ./run_my_model.py ./${FILENAME}/run_my_model.py"
cp ./run_my_model.py ./${FILENAME}/run_my_model.py
echo "cp ./my_model_helper.py ./${FILENAME}/my_model_helper.py"
cp ./my_model_helper.py ./${FILENAME}/my_model_helper.py
echo "cp ./my_model_config.py ./${FILENAME}/my_model_config.py"
cp ./my_model_config.py ./${FILENAME}/my_model_config.py
echo "cp ./my_model_plot.py ./${FILENAME}/my_model_plot.py"
cp ./my_model_plot.py ./${FILENAME}/my_model_plot.py

echo "mv ./${FILENAME} ./result/${FILENAME}"
mv ./${FILENAME} ./result/${FILENAME}