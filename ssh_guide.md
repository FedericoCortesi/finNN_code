# How can I connect to a compute node?
1. Open CMD and login with `[USER]@orcd-login004.mit.edu`, the number in the login can go from 1 to 4 and the higher number are usually faster.
    - `sinfo -p mit_normal_gpu -O Partition,Nodes,CPUs,Memory,Gres` check availability
2. Open CMD and write `salloc -p mit_normal_gpu --gres=gpu:h200:1 --cpus-per-task=8 --mem=32G --time=06:00:00` with the desired parameters.
    - The system then assigns you a node
3. Open ssh config file (from vscode) and change the parameters fro the compute node with the new assigned node.
4. Connect to orcd-compute in vscode.

> Never close the cmd where you did the login!

# How do i check the availability of the GPUs?
1. Write `nvidia-smi` in the terminal and check the output.
2. Run `gpu_test.py` and if there is no gpu run in the terminal the following code:

```
export LD_LIBRARY_PATH="$(
python - <<'PY'
import os, glob, site
paths=set()
for sp in site.getsitepackages()+[site.getusersitepackages()]:
    if not sp: continue
    for pat in ("nvidia/*/lib","nvidia/*/lib/*"):
        for d in glob.glob(os.path.join(sp, pat)):
            if os.path.isdir(d): paths.add(d)
print(":".join(sorted(paths)))
PY
):${LD_LIBRARY_PATH}
```


3. GPUs should be available in TF now. 