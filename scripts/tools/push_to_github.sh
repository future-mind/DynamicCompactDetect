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

# Check if the remote repository already exists and has content
echo "Checking remote repository..."
if git ls-remote --exit-code origin &>/dev/null; then
    echo "Remote repository exists. Attempting to integrate remote changes..."
    
    # Fetch remote content
    git fetch origin
    
    # Check if remote has a main branch
    if git ls-remote --exit-code origin main &>/dev/null; then
        remote_branch="main"
    elif git ls-remote --exit-code origin master &>/dev/null; then
        remote_branch="master"
    else
        remote_branch=""
    fi
    
    if [ -n "$remote_branch" ]; then
        echo "Remote has a '$remote_branch' branch."
        
        # Offer options to the user
        echo "The remote repository already has content. You have several options:"
        echo "1. Merge remote changes with your local changes (recommended)"
        echo "2. Force push your changes (will overwrite remote history)"
        echo "3. Create a new branch for your changes"
        echo "4. Abort push operation"
        
        read -p "Select an option (1-4): " merge_option
        
        case $merge_option in
            1)
                echo "Attempting to merge remote changes..."
                # Pull with --allow-unrelated-histories to merge unrelated repositories
                git pull origin $remote_branch --allow-unrelated-histories
                
                # Check if there were merge conflicts
                if [ $? -ne 0 ]; then
                    echo "Merge conflicts detected. Please resolve them manually and then push."
                    exit 1
                fi
                
                # Push the merged result
                git push -u origin $remote_branch
                ;;
            2)
                echo "Force pushing your changes to $remote_branch..."
                read -p "Are you sure? This will OVERWRITE remote history! (y/n): " confirm
                if [ "$confirm" = "y" ]; then
                    git push -f -u origin $remote_branch
                else
                    echo "Force push canceled."
                    exit 1
                fi
                ;;
            3)
                read -p "Enter name for new branch: " new_branch
                echo "Creating new branch '$new_branch'..."
                git checkout -b $new_branch
                git push -u origin $new_branch
                ;;
            4)
                echo "Push operation aborted."
                exit 0
                ;;
            *)
                echo "Invalid option selected. Push operation aborted."
                exit 1
                ;;
        esac
    else
        echo "Remote repository exists but doesn't have main or master branch."
        echo "Attempting to push to main branch..."
        git push -u origin main
    fi
else
    echo "Remote repository appears to be empty. Pushing to main branch..."
    # Push to GitHub (try main branch first, fall back to master if necessary)
    git push -u origin main || git push -u origin master
fi

echo "Repository push complete!"
echo "Your project is now available at: https://github.com/$github_username/DynamicCompactDetect" 