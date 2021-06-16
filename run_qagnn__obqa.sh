export CUDA_VISIBLE_DEVICES=0,1
dt=`date '+%Y%m%d_%H%M%S'`


dataset="obqa"
model='roberta-large'
if [[ "$#" -gt 0 ]]; then shift; fi
if [[ "$#" -gt 0 ]]; then shift; fi
args=$@


elr="1e-5"
dlr="1e-3"
bs=128
n_epochs=70

k=5 #num of gnn layers
gnndim=200

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "enc_name: $model"
echo "batch_size: $bs"
echo "learning_rate: elr $elr dlr $dlr"
echo "gnn: dim $gnndim layer $k"
echo "******************************"

data_root="${AMLT_DATA_DIR:-data}"
save_dir_pref="${AMLT_OUTPUT_DIR:-saved_models}"
mkdir -p $save_dir_pref

###### Training ######
for seed in 0; do
  python3 -u qagnn.py --dataset $dataset \
      --encoder $model -k $k --gnn_dim $gnndim -elr $elr -dlr $dlr -bs $bs --seed $seed \
      --n_epochs $n_epochs --max_epochs_before_stop 30  \
      --train_adj ${data_root}/${dataset}/graph/train.graph.adj.pk \
      --dev_adj   ${data_root}/${dataset}/graph/dev.graph.adj.pk \
      --test_adj  ${data_root}/${dataset}/graph/test.graph.adj.pk \
      --train_statements ${data_root}/${dataset}/statement/train.statement.jsonl \
      --dev_statements   ${data_root}/${dataset}/statement/dev.statement.jsonl \
      --test_statements  ${data_root}/${dataset}/statement/test.statement.jsonl \
      --save_model \
      --save_dir ${save_dir_pref}/enc-${model}__k${k}__gnndim${gnndim}__bs${bs}__seed${seed}__${dt} $args
done
