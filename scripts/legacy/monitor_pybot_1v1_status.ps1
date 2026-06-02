param(
    [Parameter(Mandatory = $true)]
    [string] $CorsairDir,

    [Parameter(Mandatory = $true)]
    [string] $StatusLogPath
)

$ErrorActionPreference = "Continue"
Set-Location $CorsairDir

Write-Host "TillBot 1v1 Status" -ForegroundColor Cyan
Write-Host "Waiting for bot/queue events..." -ForegroundColor DarkGray
Write-Host ""

docker logs --since 15s -f sergeant-socket-Pybot1 2>&1 | ForEach-Object {
    $line = $_.ToString()
    $timestamp = Get-Date -Format "HH:mm:ss"

    if ($line -match "READY TO PLAY") {
        Write-Host "[$timestamp] Bot online." -ForegroundColor Green
    }
    elseif ($line -match "\[connected\] username: (.+)") {
        Write-Host "[$timestamp] Username: $($Matches[1])" -ForegroundColor Green
    }
    elseif ($line -match "\[joined\] 1v1") {
        Write-Host "[$timestamp] Bot is in the public 1v1 queue and waiting for an opponent." -ForegroundColor Yellow
    }
    elseif ($line -match "\[joined\] custom") {
        Write-Host "[$timestamp] Bot is in a custom game." -ForegroundColor Yellow
    }
    elseif ($line -match "\[game_start\] replay: ([^,]+), users: (.*)") {
        Write-Host "[$timestamp] Bot is in game. Replay: https://bot.generals.io/replays/$($Matches[1])" -ForegroundColor Cyan
        Write-Host "             Players: $($Matches[2])" -ForegroundColor DarkGray
    }
    elseif ($line -match "\[game_won\] (.+)") {
        Write-Host "[$timestamp] Game won. Replay: https://bot.generals.io/replays/$($Matches[1])" -ForegroundColor Green
    }
    elseif ($line -match "\[game_lost\] ([^,]+), killer: (.*)") {
        Write-Host "[$timestamp] Game lost. Killer: $($Matches[2]). Replay: https://bot.generals.io/replays/$($Matches[1])" -ForegroundColor Red
    }
    elseif ($line -match "\[JSON\] received") {
        Write-Host "[$timestamp] Redis command was not valid JSON: $line" -ForegroundColor Red
    }
    elseif ($line -match "\[gio_error\]|\[error_set_username\]|connect_error|disconnected") {
        Write-Host "[$timestamp] $line" -ForegroundColor Red
    }

    Add-Content -LiteralPath $StatusLogPath -Value "[$timestamp] $line"
}
