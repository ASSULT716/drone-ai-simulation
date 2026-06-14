@echo off
title Drone Simulation Launcher

if exist path_config.txt (
    set /p GIT_DIR=<path_config.txt
) else (
    echo ========================================================
    echo  최초 1회 설정: Git 프로젝트 폴더 경로를 입력해주세요.
    echo  (다음 실행부터는 자동으로 이 경로가 적용됩니다.)
    echo ========================================================
    set /p GIT_DIR="▶ 경로 입력: "
    set GIT_DIR=%GIT_DIR:"=%
    echo %GIT_DIR%>path_config.txt
)

:: ⭐ 경로 유효성 검사 (자동 초기화 기능 추가)
if not exist "%GIT_DIR%\" (
    echo.
    echo --------------------------------------------------------
    echo [오류] 입력한 경로를 찾을 수 없습니다: %GIT_DIR%
    echo 잘못된 주소가 저장되어 설정 파일을 자동으로 초기화합니다.
    echo 아무 키나 누르신 뒤, 다시 실행해서 올바른 경로를 입력해주세요.
    echo --------------------------------------------------------
    :: 💡 컴퓨터가 직접 잘못된 메모장 파일을 지워버립니다!
    del path_config.txt
    pause
    exit
)

start "" "update_preprocess.exe"
start /wait "" "DroneGame.exe"

echo --------------------------------------------------------
echo 게임이 종료되었습니다.
echo 마지막 데이터 전처리를 마무리하기 위해 3분 뒤 자동으로 종료합니다...
echo --------------------------------------------------------
timeout /t 180 /nobreak

taskkill /f /im update_preprocess.exe

if not exist "%GIT_DIR%\data\processed_data\" mkdir "%GIT_DIR%\data\processed_data\"
xcopy "data\processed_data\*.npz" "%GIT_DIR%\data\processed_data\" /Y /D

exit