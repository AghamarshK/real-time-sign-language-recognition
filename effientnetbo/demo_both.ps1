# demo_both.ps1
# PowerShell script to run both demos in sequence

Write-Host "`n1. Running DAP+MaXNet Demo...`n" -ForegroundColor Green
python demo_dap.py

Write-Host "`n`n2. Running LDA+MaXNet Demo...`n" -ForegroundColor Green
python demo_lda.py