@echo off
rem Start the modelingengine server (mesh conda env). Stop with Ctrl+C
cd /d "%~dp0"
"%USERPROFILE%\miniconda3\envs\mesh\python.exe" server.py %*
pause
