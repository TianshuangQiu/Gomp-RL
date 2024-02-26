import subprocess

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
            "HERMES1",
        ]
    )
