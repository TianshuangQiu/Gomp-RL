cp envs/physics_sim ../isaacgym/python/examples/gomp.py
cp utils/ur5.py ../isaacgym/python/examples
cp utils/render.py ../isaacgym/python/examples
mkdir ../isaacgym/python/examples/graphics_images
mkdir ../isaacgym/python/examples/meshes
mkdir ../isaacgym/python/examples/depth
mkdir ../isaacgym/python/examples/poses
export LD_LIBRARY_PATH=/home/ethantqiu/anaconda3/envs/rlgpu/lib
cp -r assets ../isaacgym/

