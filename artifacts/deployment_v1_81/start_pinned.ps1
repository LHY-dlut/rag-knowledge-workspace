param(
    [ValidatePattern('^[a-z0-9][a-z0-9_-]+$')][string]$ProjectName = 'zhixu-rag-v81',
    [ValidateRange(1024,65535)][int]$Port = 18786,
    [string]$EnvFile,
    [ValidatePattern('^\d{1,3}\.\d{1,3}\.\d{1,3}\.0/24$')][string]$NetworkSubnet,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$ragSourceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$envPath = if ($EnvFile) { (Resolve-Path -LiteralPath $EnvFile).Path } else { Join-Path $ragSourceRoot '.env' }
if (-not (Test-Path -LiteralPath $envPath)) {
    throw '请先在源码根目录运行 python scripts/setup_env.py --mode production，再编辑生成的 .env。此脚本不创建或覆盖密钥。'
}
$keys = @('APP_MODE','MODEL_PROVIDER','JWT_SECRET','MYSQL_PASSWORD','MYSQL_ROOT_PASSWORD','POSTGRES_PASSWORD','DASHSCOPE_API_KEY','DASHSCOPE_HTTP_BASE_URL','DASHSCOPE_CHAT_BASE_URL','ENABLE_OCR','ENABLE_REGISTRATION','RAG_HTTP_PORT','RAG_DOCKER_SUBNET','MODEL_CONCURRENCY','MODEL_DAILY_DISPATCH_LIMIT','MODEL_MAX_OUTPUT_TOKENS')
$previous = @{}
foreach ($key in $keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue
}
try {
    [Environment]::SetEnvironmentVariable('RAG_HTTP_PORT', [string]$Port, 'Process')
    if ($NetworkSubnet) { [Environment]::SetEnvironmentVariable('RAG_DOCKER_SUBNET', $NetworkSubnet, 'Process') }
    Push-Location -LiteralPath $ragSourceRoot
    try {
        $composeArgs = @('compose','--env-file',$envPath,'-p',$ProjectName,'-f','compose.yml','-f','artifacts/deployment_v1_81/compose.pinned.yml')
        if ($NetworkSubnet) { $composeArgs += @('-f','artifacts/deployment_v1_81/compose.local-network.yml') }
        & docker @composeArgs config --quiet
        if ($LASTEXITCODE -ne 0) { throw 'Compose 配置校验失败；未启动服务。' }
        if ($ValidateOnly) { Write-Output 'Compose 配置校验通过；未启动服务。'; return }
        & docker @composeArgs up --build -d --scale worker=2
        if ($LASTEXITCODE -ne 0) { throw '服务启动失败；保留现有容器和数据库卷，请检查脱敏日志。' }
        Write-Output ('服务入口：http://127.0.0.1:' + $Port)
    } finally { Pop-Location }
} finally {
    foreach ($key in $keys) {
        if ($null -eq $previous[$key]) {
            Remove-Item -LiteralPath ('Env:' + $key) -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
        }
    }
}
