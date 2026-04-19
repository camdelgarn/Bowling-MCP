param(
    [string]$Method = "combined",
    [string]$Dir = "video\behind",
    [int]$MaxFrames = 600
)

Write-Host "Running detect_ball_image.py with method=$Method dir=$Dir max_frames=$MaxFrames"

$py = "python"
$script = "backend/detect_ball_image.py"
$args = @($script, "--dir", $Dir, "--method", $Method, "--max-frames", $MaxFrames)

Write-Host "Starting process..."
# Run and wait; output will stream to the console
& $py @args
$rc = $LASTEXITCODE
if ($rc -eq 0) {
    Write-Host "Process completed successfully."
} else {
    Write-Host "Process exited with code $rc"
    exit $rc
}
