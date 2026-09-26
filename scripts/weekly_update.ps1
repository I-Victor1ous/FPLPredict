# Windows Task Scheduler helper — runs weekly pipeline inside WSL.
param(
    [switch]$SkipScrape
)

$RepoPath = wsl.exe wslpath -a (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$Cmd = "cd '$RepoPath' && ./scripts/weekly_update.sh"
if ($SkipScrape) { $Cmd += " --skip-scrape" }
$Args = @("-d", "Ubuntu", "-e", "bash", "-lc", $Cmd)
wsl.exe @Args
