# immagine da usare
FROM python:3.7

# directory di lavoro (se non esiste viene creata)
WORKDIR /app

# installazione dipendenze e pacchetti che servono per la web app
RUN pip install pandas scikit-learn flask gunicorn statsmodels scipy matplotlib seaborn pyyaml pickle5 plotly dash

# copio nell'immagine docker i file che serviranno per esporre il modello/grafico
ADD ./model ./model
ADD ./inputs ./inputs
ADD ./templates ./templates
ADD server.py server.py
ADD model_fit.py model_fit.py
ADD my_utilities.py my_utilities.py
ADD custom_functions.py custom_functions.py
ADD plotting.py plotting.py
ADD conf.yml conf.yml

# porta che espongo all'esterno del container (porta aperta del container quindi )
EXPOSE 5000

# comando che viene lanciato quando viene runnato il container
CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "server:app" ]]