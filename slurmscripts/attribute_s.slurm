#!/bin/bash
#SBATCH --job-name=fatezero    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=az4244@princeton.edu

module purge
module load anaconda3/2023.3
conda activate avtav

accelerate launch test_fatezero.py --config="config/attribute/squ_carrot_robot_eggplant.yaml"
accelerate launch test_fatezero.py --config="config/attribute/swan_swa.yaml"



# config/attribute/bear_tiger_lion_leopard.yaml config/attribute/bus_gpu.yaml config/attribute/cat_tiger_leopard_grass.yaml config/attribute/dog_robotic_corgi.yaml config/attribute/duck_rubber.yaml config/attribute/fox_wolf_snow.yaml config/attribute/rabbit_straberry_leaves_flowers.yaml config/attribute/squ_carrot_robot_eggplant.yaml config/attribute/swan_swa.yaml