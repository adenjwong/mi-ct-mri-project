@echo off
setlocal enabledelayedexpansion

set PYTHON=python

set CASES=case01 case02 case03
set METRICS=mattes_mi joint_hist_mi
set BINS=16 32 64
set SEEDS=0 1 2 3 4

set ITERATIONS=200
set NORMALIZATION=0_1

for %%C in (%CASES%) do (
    echo ============================================================
    echo Starting workflow for %%C
    echo ============================================================

    set CT_PATH=data\%%C\ct.nii.gz
    set MRI_PATH=data\%%C\mri.nii.gz
    set OUTDIR=results\%%C_experiment
    set ANALYSIS_DIR=!OUTDIR!\analysis

    if not exist "!CT_PATH!" (
        echo Missing CT file: !CT_PATH!
        exit /b 1
    )

    if not exist "!MRI_PATH!" (
        echo Missing MRI file: !MRI_PATH!
        exit /b 1
    )

    echo Running experiments.py for %%C...
    %PYTHON% src\experiments.py ^
        --ct "!CT_PATH!" ^
        --mri "!MRI_PATH!" ^
        --outdir "!OUTDIR!" ^
        --metrics %METRICS% ^
        --bins %BINS% ^
        --seeds %SEEDS% ^
        --iterations %ITERATIONS% ^
        --normalization %NORMALIZATION% ^
        --perturb_init

    echo Running analyze_results.py for %%C...
    %PYTHON% src\analyze_results.py ^
        --summary_csv "!OUTDIR!\summary.csv" ^
        --outdir "!ANALYSIS_DIR!"

    echo Running plot_convergence_groups.py for %%C...
    %PYTHON% src\plot_convergence_groups.py ^
        --experiment_dir "!OUTDIR!" ^
        --outdir "!ANALYSIS_DIR!"

    echo Finished workflow for %%C
    echo.
)

echo All cases completed successfully.
pause