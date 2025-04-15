prob with venv :

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\venv\Scripts\Activate.ps1

to disable:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Default


no prob:

python -m venv venv
cd venv
venv\Scripts\activate

to close venv:
deactivate


install all :
pip install -r requirements.txt

launch:
python manage.py runserver




