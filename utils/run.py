import subprocess

for i in range(200):
    if i % 2 == 1:
        subprocess.run(
            [
                "python3",
                "envs/physics_sim.py",
                "--headless",
                "--num_threads",
                "6",
                "--sim_device",
                "cuda:0",
            ]
        )
    else:
        subprocess.run(
            [
                "python3",
                "envs/physics_sim.py",
                "--headless",
                "--num_threads",
                "6",
                "--sim_device",
                "cuda:1",
            ]
        )
