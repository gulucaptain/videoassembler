import argparse
import os

# parse arguments
parser = argparse.ArgumentParser(description="Launcher for ModelArts")
parser.add_argument(
    "--script_path",
    required=True,
    type=str,
    default="VATMM/scripts/runs/run1.sh",
    help="Shell script path to evaluate.",
)
parser.add_argument(
    "--world_size",
    default=1,
    type=int,
    help="Number of nodes.",
)
parser.add_argument(
    "--rank",
    default=0,
    type=int,
    help="Node rank.",
)
args, _ = parser.parse_known_args()

# get base directory
JOB_DIR = os.getenv("MA_JOB_DIR", "/home/ma-user/modelarts/user-job-dir/")

# get absolute path of script
script_path = os.path.join(JOB_DIR, args.script_path)

# test if script exist
if not os.path.exists(script_path):
    raise FileNotFoundError(script_path)

# run script
os.system(
    f"/bin/bash {script_path}"
    f" --world_size {args.world_size}"
    f" --node_rank {args.rank}"
)
