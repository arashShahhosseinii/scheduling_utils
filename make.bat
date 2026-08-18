@echo off
setlocal

if "%1"=="" goto help

if /I "%1"=="generate" goto generate
if /I "%1"=="validate" goto validate
if /I "%1"=="pilot" goto pilot
if /I "%1"=="train" goto train
if /I "%1"=="evaluate" goto evaluate
if /I "%1"=="tensorboard" goto tensorboard
if /I "%1"=="clean" goto clean

echo Unknown command: %1
goto help

:generate
python gen_dataset.py
if errorlevel 1 exit /b 1
exit /b 0

:validate
python validate_gang.py
if errorlevel 1 exit /b 1
exit /b 0

:pilot
python train_ppo_gat.py --timesteps 20000
if errorlevel 1 exit /b 1
exit /b 0

:train
python train_ppo_gat.py --timesteps 200000
if errorlevel 1 exit /b 1
exit /b 0

:evaluate
python main.py
if errorlevel 1 exit /b 1
exit /b 0

:tensorboard
python -m tensorboard.main --logdir artifacts_gang\logs\tensorboard
exit /b 0

:clean
if exist artifacts_gang rmdir /s /q artifacts_gang
echo Removed artifacts_gang.
exit /b 0

:help
echo.
echo Usage: .\make.bat COMMAND
echo.
echo Commands:
echo   generate    Generate dag_dataset_gang_a7_a12.csv from extracted results____
echo   validate    Run check_env + HEFT_GANG scheduling invariants
echo   pilot       Run a 20,000-step PPO+GAT+Gang pilot
echo   train       Run the full 200,000-step training
echo   evaluate    Evaluate PPO_GANG vs HEFT_GANG
echo   tensorboard Open TensorBoard logs
echo   clean       Remove ONLY artifacts_gang; old artifacts remain untouched
echo.
exit /b 0
