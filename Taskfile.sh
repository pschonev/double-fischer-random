#!/bin/bash  

# Fetch a new position to analyze  
fetch_position() {  
  echo "Fetching a new position to analyze..."  
  GITHUB_USER=$(git config user.name)
  POSITION=$(python fetch_position.py --github-user "$GITHUB_USER")  
}  

run_analysis() {  
  POSITION=$1  
  echo "Running analysis on position: $POSITION"  
  python run_analysis.py --position "$POSITION"  
}  

# Push results to a PR  
push_results() {  
  echo "Pushing results to the PR..."  
  git add .wip_analysis.json 
  git commit -m "Add analysis results for $1"  
  git push origin HEAD

  if grep -q '"validator":' .wip_analysis.json; then
    gh pr edit --add-label validated --remove-label validation_in_progress
  else
    gh pr edit --add-label analyzed --remove-label analysis_in_progress
  fi
}

# Combined command to fetch, analyze, and push  
analyze() {  
  fetch_position  
  run_analysis "$POSITION"  
  push_results "$POSITION"  
}  

# Check the command line argument and call the appropriate function  
case "$1" in  
  fetch_position)  
    fetch_position  
    ;;  
  run_analysis)  
    run_analysis "$2"  
    ;;  
  push_results)  
    push_results "$2"  
    ;;  
  analyze)  
    analyze  
    ;;  
  *)  
    echo "Usage: $0 {fetch_position|run_analysis|push_results|analyze}"  
    exit 1  
    ;;  
esac
