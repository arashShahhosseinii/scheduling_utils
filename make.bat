@echo off
setlocal enabledelayedexpansion

set COMMAND=%1

if "%COMMAND%"=="" (
    set COMMAND=run
)

set ARTIFACT_DIR=artifacts
set MODEL_DIR=artifacts\models
set EVAL_DIR=artifacts\evaluation
set LOG_DIR=artifacts\logs
set MODEL_FILE=artifacts\models\ppo_mlp_scheduler.zip

if "%COMMAND%"=="run" goto run
if "%COMMAND%"=="install" goto install
if "%COMMAND%"=="check" goto check
if "%COMMAND%"=="train" goto train
if "%COMMAND%"=="evaluate" goto evaluate
if "%COMMAND%"=="quick" goto quick
if "%COMMAND%"=="results" goto results
if "%COMMAND%"=="clean" goto clean

echo Unknown command: %COMMAND%
echo.
echo Available commands:
echo   make run
echo   make install
echo   make check
echo   make train
echo   make evaluate
echo   make quick
echo   make results
echo   make clean
exit /b 1


:check
echo.
echo ============================================================
echo  Checking required files
echo ============================================================

if not exist requirements.txt (
    echo ERROR: Missing requirements.txt
    exit /b 1
)

if not exist train_ppo_gat.py (
    echo ERROR: Missing train_ppo_gat.py
    exit /b 1
)

if not exist main.py (
    echo ERROR: Missing main.py
    exit /b 1
)

if not exist Dag_Env.py (
    echo ERROR: Missing Dag_Env.py
    exit /b 1
)

if not exist Create_Dag.py (
    echo ERROR: Missing Create_Dag.py
    exit /b 1
)

if not exist gat_sb3_policy.py (
    echo ERROR: Missing gat_sb3_policy.py
    exit /b 1
)

if not exist scheduling_utils.py (
    echo ERROR: Missing scheduling_utils.py
    exit /b 1
)

if not exist "dag_dataset1_a7_a12.csv" (
    echo ERROR: Missing dag_dataset1_a7_a12.csv
    exit /b 1
)

if not exist "dag_dataset2_a7_a12.csv" (
    echo ERROR: Missing dag_dataset2_a7_a12.csv
    exit /b 1
)

if not exist "dag_dataset3_a7_a12.csv" (
    echo ERROR: Missing dag_dataset3_a7_a12.csv
    exit /b 1
)

echo All required files exist.
exit /b 0


:install
echo.
echo ============================================================
echo  Installing dependencies
echo ============================================================

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    exit /b 1
)

echo Dependencies installed successfully.
exit /b 0


:train
call "%~f0" check
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo  Training PPO + GAT model
echo ============================================================

if not exist "%ARTIFACT_DIR%" mkdir "%ARTIFACT_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

python train_ppo_gat.py
if errorlevel 1 (
    echo ERROR: Training failed.
    exit /b 1
)

echo.
echo Training finished.
echo Model path:
echo %MODEL_FILE%
exit /b 0


:evaluate
call "%~f0" check
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo  Evaluating PPO+GAT against HEFT
echo ============================================================

if not exist "%MODEL_FILE%" (
    echo ERROR: Model not found: %MODEL_FILE%
    echo First run:
    echo   make train
    exit /b 1
)

if not exist "%EVAL_DIR%" mkdir "%EVAL_DIR%"

python main.py
if errorlevel 1 (
    echo ERROR: Evaluation failed.
    exit /b 1
)

echo.
echo Evaluation finished.
exit /b 0


:results
echo.
echo ============================================================
echo  Output files
echo ============================================================

if exist "%EVAL_DIR%\comparison_detail.csv" (
    echo Found: %EVAL_DIR%\comparison_detail.csv
) else (
    echo Not found: %EVAL_DIR%\comparison_detail.csv
)

if exist "%EVAL_DIR%\comparison_summary.csv" (
    echo Found: %EVAL_DIR%\comparison_summary.csv
) else (
    echo Not found: %EVAL_DIR%\comparison_summary.csv
)

if exist "%EVAL_DIR%\makespan_comparison.png" (
    echo Found: %EVAL_DIR%\makespan_comparison.png
) else (
    echo Not found: %EVAL_DIR%\makespan_comparison.png
)

if exist "%EVAL_DIR%\energy_comparison.png" (
    echo Found: %EVAL_DIR%\energy_comparison.png
) else (
    echo Not found: %EVAL_DIR%\energy_comparison.png
)

echo.

if exist "%EVAL_DIR%\comparison_summary.csv" (
    echo Summary CSV content:
    echo ------------------------------------------------------------
    type "%EVAL_DIR%\comparison_summary.csv"
    echo.
    echo ------------------------------------------------------------
)

exit /b 0


:quick
call "%~f0" evaluate
if errorlevel 1 exit /b 1

call "%~f0" results
if errorlevel 1 exit /b 1

echo.
echo Quick evaluation completed successfully.
exit /b 0


:clean
echo.
echo ============================================================
echo  Cleaning artifacts
echo ============================================================

if exist "%ARTIFACT_DIR%" (
    rmdir /s /q "%ARTIFACT_DIR%"
    echo Removed: %ARTIFACT_DIR%
) else (
    echo No artifacts directory found.
)

echo Clean completed.
exit /b 0


:run
call "%~f0" install
if errorlevel 1 exit /b 1

call "%~f0" train
if errorlevel 1 exit /b 1

call "%~f0" evaluate
if errorlevel 1 exit /b 1

call "%~f0" results
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo  Full project run completed successfully.
echo ============================================================
exit /b 0