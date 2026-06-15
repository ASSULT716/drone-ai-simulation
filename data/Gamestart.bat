@echo off
title Drone Simulation Launcher

:: 1. 백그라운드 전처리 엔진 가동
start "" "update_preprocess.exe"

:: 2. 실제 게임 실행 파일 구동 및 유저가 끌 때까지 대기
start /wait "" "drone_simulation.exe"

:: 3. 게임이 꺼진 후, 남은 전처리를 마무리하도록 3분(180초) 대기
echo --------------------------------------------------------
echo Game play is finished.
echo Waiting 5 minites to preprocess data file.
echo If preprocess is done, press ctrl+c and y sequentially to close bat file.
echo --------------------------------------------------------
timeout /t 300 /nobreak

:: 4. 5분이 지나면 전처리 엔진 차단
taskkill /f /im update_preprocess.exe

exit