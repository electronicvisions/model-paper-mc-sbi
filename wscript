import os
from os.path import join
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path
from waflib import Utils


EXPERIMENT_NAME: str = "paper_sbi"

def depends(ctx):
    ctx("pynn-brainscales")
    ctx("calix")
    ctx("model-hw-mc-genetic")
    ctx("code-format")


def options(opt):
    opt.load("test_base")
    opt.load("pytest")


def configure(conf):
    conf.load("test_base")
    conf.load("pytest")

    conf.load("python")


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    bld(name=f"{EXPERIMENT_NAME}-python_libraries",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob("src/py/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/lib",
        install_from="src/py",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        use=["pynn_brainscales2", "model_hw_mc_genetic-python_libraries",
		     "calix_pylib"],
        test_timeout=120)

    bld(name=f"{EXPERIMENT_NAME}-python_scripts",
        features="py use pylint pycodestyle",
        source=bld.path.ant_glob(f"src/py/{EXPERIMENT_NAME}/scripts/**/*.py"),
        relative_trick=True,
        install_path="${PREFIX}/bin",
        install_from=f"src/py/{EXPERIMENT_NAME}/scripts",
        chmod=Utils.O755,
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        use=["model_hw_mc_genetic-python_libraries", "pynn_brainscales2",
		     f"{EXPERIMENT_NAME}-python_libraries", "calix_pylib"],
        test_timeout=120)

    bld(name=f"{EXPERIMENT_NAME}-python_hwtests",
        tests=bld.path.ant_glob("tests/hw/py/**/*.py"),
        features="use pytest pylint pycodestyle",
        use=[f"{EXPERIMENT_NAME}-python_libraries", "model_hw_mc_genetic-python_libraries"],
        install_path="${PREFIX}/bin/tests/hw",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE,
        test_timeout=120)

    bld(name=f"{EXPERIMENT_NAME}-python_swtests",
        tests=bld.path.ant_glob("tests/sw/py/**/*.py"),
        features="use pytest pylint pycodestyle",
        use=[f"{EXPERIMENT_NAME}-python_libraries", "model_hw_mc_genetic-python_libraries"],
        install_path="${PREFIX}/bin/tests/sw",
        pylint_config=join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=join(get_toplevel_path(), "code-format", "pycodestyle"),
        test_timeout=160)

    bld.add_post_fun(summary)
