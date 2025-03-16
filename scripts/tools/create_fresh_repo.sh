#!/bin/bash
# Script to create a fresh GitHub repository 

echo "=== Create Fresh GitHub Repository ==="
echo "This script will help you create a completely fresh repository."
echo "WARNING: This will DELETE any existing repository with the same name!"
echo ""

# Ask for GitHub username
read -p "Enter your GitHub username (default: future-mind): " github_username
github_username=${github_username:-future-mind}

# Ask for repository name
read -p "Enter repository name (default: DynamicCompactDetect): " repo_name
repo_name=${repo_name:-DynamicCompactDetect}

echo ""
echo "This will create a fresh repository at: https://github.com/$github_username/$repo_name"
echo "WARNING: If this repository already exists, it will be DELETED!"
read -p "Are you absolutely sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

# Check for GitHub CLI (gh)
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. You'll need to manually delete the existing repository."
    echo "Please visit https://github.com/$github_username/$repo_name and delete it from the Settings tab."
    read -p "Have you deleted the repository? (y/n): " deleted
    if [ "$deleted" != "y" ]; then
        echo "Please delete the repository before continuing."
        exit 1
    fi
else
    # Use GitHub CLI to delete repository if it exists
    echo "Checking if repository exists..."
    if gh repo view "$github_username/$repo_name" &>/dev/null; then
        echo "Repository exists. Attempting to delete it..."
        gh repo delete "$github_username/$repo_name" --yes
        # Wait a bit for GitHub to process the deletion
        echo "Waiting for GitHub to process the deletion..."
        sleep 5
    else
        echo "Repository does not exist. Proceeding with creation."
    fi
fi

# Create new repository on GitHub
echo "Creating new repository: $github_username/$repo_name"
if command -v gh &> /dev/null; then
    gh repo create "$github_username/$repo_name" --public --description "DynamicCompactDetect: A lightweight object detection model" --confirm
else
    echo "Please create a new repository on GitHub:"
    echo "1. Go to https://github.com/new"
    echo "2. Set repository name to: $repo_name"
    echo "3. Set visibility to: Public"
    echo "4. Do NOT initialize with a README, .gitignore, or license"
    read -p "Have you created the repository? (y/n): " created
    if [ "$created" != "y" ]; then
        echo "Please create the repository before continuing."
        exit 1
    fi
fi

# Configure local git repository
echo "Configuring local git repository..."
# Remove existing .git directory if it exists
if [ -d ".git" ]; then
    rm -rf .git
fi

# Initialize new git repository
git init
git add .
git commit -m "Initial commit of DynamicCompactDetect"

# Add GitHub remote
git remote add origin "https://github.com/$github_username/$repo_name.git"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main || git push -u origin master

echo ""
echo "Repository setup complete!"
echo "Your fresh repository is now available at: https://github.com/$github_username/$repo_name" 