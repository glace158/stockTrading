import os
path = os.path.dirname(__file__)  + "/PPO_preTrained/Richdog/PPO_Richdog_0_20250409-161810.pth"
file_name, file_extension = os.path.splitext(os.path.basename(path))
run_path = os.getcwd()

print(os.path.dirname(path))

relative_path = os.path.relpath(path, run_path)
print(relative_path)

print(file_name.split('_')[-1])
dirname, basename = os.path.split(relative_path)
print(dirname.split('/')[-1])
print(basename)

print(next(os.walk(dirname))[2])