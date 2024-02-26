import subprocess

while True:
    subprocess.run(
        [
            "python3",
            "envs/physics_sim.py",
            "--headless",
            "--num_threads",
            "6",
            "--sim_device",
            "cuda:1",
            "--num_envs",
            "60",
        ]
    )
