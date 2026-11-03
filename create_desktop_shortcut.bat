@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "$WshShell = New-Object -ComObject WScript.Shell; $Desktop = [Environment]::GetFolderPath('Desktop'); $Shortcut = $WshShell.CreateShortcut(\"$Desktop\UniView.lnk\"); $Shortcut.TargetPath = 'D:\Conda\envs\unimate\pythonw.exe'; $Shortcut.Arguments = '\"d:\UniView\main.py\"'; $Shortcut.WorkingDirectory = 'd:\UniView'; $Shortcut.Description = 'UniView - 3D Asset & Rig Visualizer'; $Shortcut.Save()"
echo Shortcut UniView created on Desktop!
pause
