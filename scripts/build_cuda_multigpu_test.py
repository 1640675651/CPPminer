"""Link the multi-GPU integration harness against a Unix Makefiles miner build."""
import argparse
import pathlib
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=pathlib.Path, required=True)
parser.add_argument('--output', type=pathlib.Path)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
build = args.build.resolve()
output = (args.output or build / 'cp_cuda_multigpu_test').resolve()
directory = build / 'CMakeFiles/cppminer.dir'
flags = (directory / 'flags.make').read_text()

def options(name):
    return shlex.split(flags.split(name + ' = ', 1)[1].splitlines()[0])

link = shlex.split((directory / 'link.txt').read_text())
if not any('libcp_proof_ffi.a' in entry for entry in link):
    raise SystemExit('A real Rust proof FFI build is required; proof stubs cannot validate this test.')
compiler = (directory / 'build.make').read_text()
cuda_command = next(line.strip() for line in compiler.splitlines()
                    if 'nvcc' in line and '-c ' in line and 'cp_gpu.cu' in line)
nvcc = shlex.split(cuda_command)[0]
test_object = output.with_suffix('.o')
output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([nvcc, *options('CUDA_DEFINES'), *options('CUDA_INCLUDES'),
                *options('CUDA_FLAGS'), '-c', str(root / 'tests/cp_cuda_multigpu_test.cu'),
                '-o', str(test_object)], cwd=build, check=True)
link = [entry for entry in link
        if not entry.endswith(('src/cuda/cp_gpu.cu.o', 'src/common/main.cpp.o'))]
link[link.index('-o') + 1] = str(output)
link.insert(link.index('-o'), str(test_object))
subprocess.run(link, cwd=build, check=True)
print(output)
