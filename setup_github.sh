#!/bin/bash
# GitHub repository setup script

# Initialize Git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit: Sales Conversion Analytics Dashboard"

# Set remote repository
git remote add origin https://github.com/domtom5d/sales-dashboard.git

# Push to GitHub
git branch -M main
git push -u origin main

echo "Repository successfully pushed to GitHub!"
echo "View your project at: https://github.com/domtom5d/sales-dashboard"