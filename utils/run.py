import subprocess
import os
import shutil

counter = 0
prefix = "HERMES1"
os.makedirs("BATCHED", exist_ok=True)
while True:
    subprocess.run(
        [
            "python3",
            "envs/physics_sim.py",
            "--headless",
            "--sim_device",
            "cuda:0",
            "--num_envs",
            "4",
            "--prefix",
            prefix,
        ]
    )
    counter += 1
    if counter % 100 == 0:
        counter = 0
        total_amount = len(os.listdir("BATCHED")) + 1
        os.makedirs(f"BATCHED/{total_amount}/poses", exist_ok=True)
        os.makedirs(f"BATCHED/{total_amount}/depth", exist_ok=True)
        for file in os.listdir(f"{prefix}/depth"):
            shutil.move(f"{prefix}/depth/{file}", f"BATCHED/{total_amount}/depth/{file}")
        for file in os.listdir(f"{prefix}/poses"):
            shutil.move(f"{prefix}/poses/{file}", f"BATCHED/{total_amount}/poses/{file}")
