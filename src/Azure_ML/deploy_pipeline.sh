# Execute this script with . ./deploy_pipeline.sh !!!
cd ../
mv .gitignore .gitignoreorig
mv .gitignoredeploy .gitignore

/Users/andreasditte/Desktop/Private_Projekte/Sonntagsfrage/src/.venv/bin/python /Users/andreasditte/Desktop/Private_Projekte/Sonntagsfrage/src/Azure_ML/03_pipeline/setup.py --no-trigger-after-publish --no-schedule

mv .gitignore .gitignoredeploy
mv .gitignoreorig .gitignore

cd Azure_ML/
