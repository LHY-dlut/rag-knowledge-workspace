param(
    [ValidateSet('demo', 'dashscope')][string]$Models = 'demo',
    [ValidateSet('api', 'worker')][string]$Role = 'api',
    [string]$CredentialCsv = '',
    [int]$Port = 18081
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install requirements.lock first.' }
$env:PYTHONUTF8 = '1'
$env:APP_MODE = 'demo'
$env:MODEL_PROVIDER = $Models
$env:ENABLE_API_WORKERS = 'false'
$env:BUSINESS_DB_URL = "sqlite://data/local-$Models/business.sqlite3"
$env:VECTOR_DB_URL = "sqlite://data/local-$Models/vectors.sqlite3"
$env:UPLOAD_DIR = "data/local-$Models/uploads"
if ($Models -eq 'dashscope') {
    if (-not (Test-Path -LiteralPath $CredentialCsv)) { throw 'Provide -CredentialCsv with the local exported credentials file.' }
    $rows = Import-Csv -LiteralPath $CredentialCsv -Header 'Name','Value'
    $values = @{}
    foreach ($row in $rows) { $values[$row.Name] = $row.Value }
    $env:DASHSCOPE_API_KEY = $values['apiKey']
    $env:DASHSCOPE_HTTP_BASE_URL = $values['dashScope']
    $env:DASHSCOPE_CHAT_BASE_URL = $values['openAiCompatible']
}
try {
    if ($Role -eq 'api') {
        & $python -m app.bootstrap
        if ($LASTEXITCODE -ne 0) { throw 'Database bootstrap failed.' }
        & $python -m uvicorn app.main:app --host 127.0.0.1 --port $Port
    } else {
        & $python -m app.worker
    }
} finally {
    Remove-Item Env:DASHSCOPE_API_KEY -ErrorAction SilentlyContinue
}
