#!/bin/bash
# End-to-end test script for DynamicCompactDetect
# This script tests the full pipeline from a fresh clone

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${YELLOW}DynamicCompactDetect End-to-End Test${NC}"
echo -e "${YELLOW}=====================================================${NC}"

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo -e "${GREEN}Created temporary test directory: ${TEST_DIR}${NC}"

# Cleanup function that will be called on exit
function cleanup {
    echo -e "${YELLOW}Cleaning up test directory...${NC}"
    rm -rf "$TEST_DIR"
    echo -e "${GREEN}Cleanup complete.${NC}"
}

# Register the cleanup function to be called on exit
trap cleanup EXIT

# Step 1: Clone the repository
echo -e "${YELLOW}Step 1: Cloning the repository...${NC}"
git clone https://github.com/future-mind/dynamiccompactdetect.git "$TEST_DIR/dynamiccompactdetect"
cd "$TEST_DIR/dynamiccompactdetect"
echo -e "${GREEN}Repository cloned successfully.${NC}"

# Step 2: Setup virtual environment
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"
python3 -m venv venv
if [ "$(uname)" == "Darwin" ] || [ "$(uname)" == "Linux" ]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi
echo -e "${GREEN}Virtual environment activated.${NC}"

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# Step 4: Run basic test script
echo -e "${YELLOW}Step 4: Running basic model test...${NC}"
if [ -f "tests/test_inference.py" ]; then
    python tests/test_inference.py
    TEST_RESULT=$?
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}Basic model test passed.${NC}"
    else
        echo -e "${RED}Basic model test failed with error code $TEST_RESULT.${NC}"
        exit 1
    fi
else
    echo -e "${RED}Test script not found. Skipping basic test.${NC}"
fi

# Step 5: Run the complete pipeline
echo -e "${YELLOW}Step 5: Running complete pipeline...${NC}"
chmod +x run_dcd_pipeline.sh
./run_dcd_pipeline.sh --compare-only
PIPELINE_RESULT=$?
if [ $PIPELINE_RESULT -eq 0 ]; then
    echo -e "${GREEN}Pipeline execution successful.${NC}"
else
    echo -e "${RED}Pipeline execution failed with error code $PIPELINE_RESULT.${NC}"
    exit 1
fi

# Step 6: Generate paper data
echo -e "${YELLOW}Step 6: Generating research paper data...${NC}"
./run_dcd_pipeline.sh --paper
PAPER_RESULT=$?
if [ $PAPER_RESULT -eq 0 ]; then
    echo -e "${GREEN}Paper data generation successful.${NC}"
else
    echo -e "${RED}Paper data generation failed with error code $PAPER_RESULT.${NC}"
    exit 1
fi

# Step 7: Verify results
echo -e "${YELLOW}Step 7: Verifying results...${NC}"
# Check for comparison results
if [ -d "results/comparisons" ] && [ "$(ls -A results/comparisons)" ]; then
    echo -e "${GREEN}Comparison results found.${NC}"
else
    echo -e "${RED}Comparison results not found or empty.${NC}"
    exit 1
fi

# Check for paper data
if [ -d "results/research_paper" ] && [ "$(ls -A results/research_paper)" ]; then
    echo -e "${GREEN}Research paper data found.${NC}"
else
    echo -e "${RED}Research paper data not found or empty.${NC}"
    exit 1
fi

echo -e "${YELLOW}=====================================================${NC}"
echo -e "${GREEN}All tests passed! DynamicCompactDetect is working correctly.${NC}"
echo -e "${YELLOW}=====================================================${NC}"

exit 0 