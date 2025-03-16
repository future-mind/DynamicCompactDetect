#!/bin/bash
# Script to initialize the Git repository and push it to GitHub

# Set the base directory
BASE_DIR=$(pwd)

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Initialize git repository
echo "Initializing git repository..."
git init

# Add all files
echo "Adding files to git..."
git add .

# Commit changes
echo "Committing changes..."
git commit -m "Initial commit of DynamicCompactDetect"

# Configure git with username and email if not already done
git_username=$(git config --get user.name)
git_email=$(git config --get user.email)

if [ -z "$git_username" ]; then
    read -p "Enter your GitHub username: " username
    git config user.name "$username"
fi

if [ -z "$git_email" ]; then
    read -p "Enter your GitHub email: " email
    git config user.email "$email"
fi

# Add remote
echo "Adding remote repository..."
read -p "Enter your GitHub username (default: future-mind): " github_username
github_username=${github_username:-future-mind}

git remote add origin "https://github.com/$github_username/DynamicCompactDetect.git"

# Check if GitHub personal access token exists
token_file="$HOME/.github_token"
if [ ! -f "$token_file" ]; then
    echo "You'll need a GitHub Personal Access Token for pushing to GitHub."
    echo "You can create one at: https://github.com/settings/tokens"
    echo "Make sure to select 'repo' scope."
    read -p "Enter your GitHub Personal Access Token: " github_token
    echo "$github_token" > "$token_file"
    chmod 600 "$token_file"
fi

# Push to GitHub
echo "Pushing to GitHub..."
echo "You may be prompted for your GitHub username and password or token."
git push -u origin main || git push -u origin master

echo "Repository push complete!"
echo "Your project is now available at: https://github.com/$github_username/DynamicCompactDetect" 