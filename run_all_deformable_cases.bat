@echo off
setlocal enabledelayedexpansion

set PYTHON=python
set CASES=case01 case02 case03
set SEEDS=0 1 2

for %%C in (%CASES%) do (
    for %%S in (%SEEDS%) do (
        echo ============================================================
        echo Running deformable workflow for %%C, seed %%S
        echo ============================================================

        %PYTHON% src\run_deformable_case.py ^
            --ct data\%%C\ct.nii.gz ^
            --mri data\%%C\mri.nii.gz ^
            --outdir results\%%C_deformable_seed%%S ^
            --metric mattes_mi ^
            --bins 32 ^
            --rigid_iterations 200 ^
            --deformable_iterations 75 ^
            --perturb_init ^
            --seed %%S ^
            --mesh_x 4 ^
            --mesh_y 4 ^
            --mesh_z 3
    )
)

echo All deformable runs complete.
pause