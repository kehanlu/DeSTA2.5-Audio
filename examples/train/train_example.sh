export HF_TOKEN="hf_..."


export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="/lab/DeSTA2.5-Audio":$PYTHONPATH
export ROOT_DIR="/lab/DeSTA2.5-Audio"


config=desta25_debug
dataset_config=debug
devices=2

data_root=/lab/DeSTA2.5-Audio

project=debug
name=""
exp_name=$(date +%y%m%d-%H)@${name}
exp_dir="${ROOT_DIR}/my_exps/${project}/${exp_name}"


resume_from_checkpoint=null
init_from_pretrained_weights=null


# record git diff
mkdir -p ${exp_dir}
git diff > ${exp_dir}/git-diff.txt
nvidia-smi > ${exp_dir}/nvidia-smi.txt

python ${ROOT_DIR}/examples/train/train_desta.py \
    --config-path=config \
    --config-name=${config} \
    trainer.devices=${devices} \
    +dataset=${dataset_config} \
    +exp_dir=${exp_dir} \
    project=${project} \
    name=${name} \
    +dataset.train_ds.data_root=${data_root} \
    +dataset.validation_ds.data_root=${data_root} \
    +resume_from_checkpoint=${resume_from_checkpoint} \
    +init_from_pretrained_weights=${init_from_pretrained_weights}
