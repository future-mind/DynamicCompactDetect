# DynamicCompactDetect Pipeline Script (PowerShell version)
# This script provides the same functionality as run_dcd_pipeline.sh for Windows users

# Function to display help message
function Show-Help {
    Write-Host "DynamicCompactDetect Pipeline"
    Write-Host "Usage: .\run_dcd_pipeline.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -h, --help             Show this help message"
    Write-Host "  -c, --compare-only     Only run model comparison (skip fine-tuning)"
    Write-Host "  -o, --output-dir DIR   Set custom output directory (default: results)"
    Write-Host "  -r, --runs N           Number of inference runs per image (default: 3)"
    Write-Host "  -p, --paper            Generate research paper data"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run_dcd_pipeline.ps1                     # Run the complete pipeline"
    Write-Host "  .\run_dcd_pipeline.ps1 -compare-only       # Only run model comparison"
    Write-Host "  .\run_dcd_pipeline.ps1 -output-dir custom_results -runs 5"
    Write-Host "  .\run_dcd_pipeline.ps1 -paper              # Generate research paper data"
    exit 0
}

# Function to check Python environment
function Check-Python {
    try {
        $pythonVersion = python --version
        Write-Host "Found Python: $pythonVersion"
    }
    catch {
        Write-Host "Error: Python not found. Please install Python 3.8 or later." -ForegroundColor Red
        exit 1
    }

    # Check if virtual environment exists
    if (-not (Test-Path "venv")) {
        Write-Host "Virtual environment not found. Creating one..." -ForegroundColor Yellow
        python -m venv venv
    }

    # Activate virtual environment
    Write-Host "Activating virtual environment..."
    & .\venv\Scripts\Activate.ps1

    # Check if requirements are installed
    try {
        python -c "import ultralytics" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Ultralytics not installed"
        }
        Write-Host "Required packages are installed." -ForegroundColor Green
    }
    catch {
        Write-Host "Installing required packages..." -ForegroundColor Yellow
        pip install -r requirements.txt
    }
}

# Function to create directories
function Create-Directories {
    param (
        [string]$outputDir = "results"
    )

    $directories = @(
        $outputDir,
        "$outputDir\benchmarks",
        "$outputDir\comparisons",
        "$outputDir\research_paper",
        "$outputDir\research_paper\figures",
        "$outputDir\research_paper\tables",
        "data",
        "data\test_images",
        "models"
    )

    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Green
        }
    }
}

# Function to check for required model files
function Check-Models {
    $requiredModels = @{
        "models\yolov8n.pt" = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt";
        "models\dynamiccompactdetect_finetuned.pt" = $null
    }

    $allModelsPresent = $true

    foreach ($model in $requiredModels.Keys) {
        if (-not (Test-Path $model)) {
            $allModelsPresent = $false
            $downloadUrl = $requiredModels[$model]
            
            if ($downloadUrl) {
                Write-Host "Downloading $model from $downloadUrl..." -ForegroundColor Yellow
                Invoke-WebRequest -Uri $downloadUrl -OutFile $model
                Write-Host "Downloaded $model successfully." -ForegroundColor Green
            }
            else {
                Write-Host "Warning: $model not found and no download URL available." -ForegroundColor Yellow
                Write-Host "Please manually place the required model file in the models directory." -ForegroundColor Yellow
            }
        }
        else {
            Write-Host "Found model: $model" -ForegroundColor Green
        }
    }

    return $allModelsPresent
}

# Function to run model comparison
function Run-ModelComparison {
    param (
        [string]$outputDir = "results",
        [int]$numRuns = 3
    )

    Write-Host "Running model comparison..." -ForegroundColor Cyan
    python scripts\compare_all_models.py --num-runs $numRuns --output-dir "$outputDir\comparisons"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Model comparison failed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Model comparison completed successfully." -ForegroundColor Green
}

# Function to generate research paper data
function Generate-PaperData {
    param (
        [string]$outputDir = "results"
    )

    Write-Host "Generating research paper data..." -ForegroundColor Cyan
    python scripts\generate_paper_data.py --output-dir $outputDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Paper data generation failed." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Research paper data generated successfully." -ForegroundColor Green
}

# Parse command line arguments
$compareOnly = $false
$outputDir = "results"
$numRuns = 3
$generatePaper = $false

for ($i = 0; $i -lt $args.Count; $i++) {
    switch ($args[$i]) {
        { $_ -in "-h", "--help" } { Show-Help }
        { $_ -in "-c", "--compare-only" } { $compareOnly = $true }
        { $_ -in "-o", "--output-dir" } { 
            $i++
            if ($i -lt $args.Count) {
                $outputDir = $args[$i]
            }
        }
        { $_ -in "-r", "--runs" } { 
            $i++
            if ($i -lt $args.Count) {
                $numRuns = [int]$args[$i]
            }
        }
        { $_ -in "-p", "--paper" } { $generatePaper = $true }
    }
}

# Main execution
Write-Host "DynamicCompactDetect Pipeline (PowerShell)" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check Python environment
Check-Python

# Create necessary directories
Create-Directories -outputDir $outputDir

# Check for required model files
$modelsPresent = Check-Models

if (-not $modelsPresent) {
    Write-Host "Warning: Some model files are missing. The pipeline may not work correctly." -ForegroundColor Yellow
}

# Run the requested operations
if ($compareOnly) {
    Run-ModelComparison -outputDir $outputDir -numRuns $numRuns
}
elseif ($generatePaper) {
    Generate-PaperData -outputDir $outputDir
}
else {
    # Run the complete pipeline
    Run-ModelComparison -outputDir $outputDir -numRuns $numRuns
    Generate-PaperData -outputDir $outputDir
}

Write-Host "DynamicCompactDetect pipeline completed successfully!" -ForegroundColor Green
Write-Host "Results are available in the '$outputDir' directory." -ForegroundColor Green 