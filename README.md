# IHE Project

This is the IHE project repository. 

## Setup Guide

This repository only contains our own project code. 
The original school-provided files / datasets are **not uploaded to GitHub** because they are large and include many 
unnecessary files. 

## Install Git 

If you are using Windows, please install `Git` first: 

https://git-scm.com/download/win

After installation, open `PowerShell`, `cmd`, or terminals on any of your IDE and check: 

```Bash
git --version
```

If it shows a Git version number, Git is installed successfully. 

## 1. Clone the repository 

Open terminal, use `cmd` or `PowerShell` for `Windows`, and `terminal` for `MacOS`. 

```bash
git clone https://github.com/eliadecao/IHE-project.git
cd IHE-project
```

## 2. Add the school-provided files manually 

After cloning, manually copy the school-provided folder into the project directory. 

Expected structure: 

```text
IHE-project/
|-- Kinematics/   # copy the 'Kinematics' directory from the school-provided directory. 
|-- code/
|-- README.md
`-- .gitignore
```

The `Kinematics/` folder is already listed in `.gitignore`. so Git will ignore it automatically when pushing latest 
change to GitHub. 

## 3. Working with Git

Before making changes, remember to fetch and pull the latest version: 

```Bash
git fetch 
git pull
```

After editing the code: 

```Bash
git status
git add . 
git commit -m "Describe your change to the code. "
git push
```

## 4. Working with Branches 

Each team member should work on their own branch, for collaboration. 
Branches for classification task and regression task are created separately. 

After cloning the repository, switch to your own branch: 

```Bash
git fetch
git switch your-branch-name  # replace with `classification` or `regression`. 
git pull origin your-branch-name 
```

Note: two separate directories for both classification task and regression task are created as well, please implement 
the code under the corresponding directory. 

Please do not work directly on the `main` branch. Shall merge the two branches to `main` when both tasks are working 
correctly. 